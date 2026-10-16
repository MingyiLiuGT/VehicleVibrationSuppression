"""Two-DOF suspension, constrained preview MPC, mass RLS and bounded NN.

All coordinates are deviations from static equilibrium. Road and force are
zero-order held on each 10-ms interval. No external road data are required.
"""
import bootstrap
import numpy as np
from scipy.linalg import expm
from scipy import sparse
import osqp
from dataclasses import dataclass

DT = .01
HORIZON = 30
FORCE_SCALE = 500.
KS, CS, KT, MU = 20000., 2000., 200000., 40.
TRAVEL_SCALE = .08
LOAD_SCALE = 340 * 9.81
WEIGHTS = np.array([1., .02, .02])
EFFORT = .005

def continuous(mass=300., ks=KS, cs=CS):
    a = np.array([[0,1,0,0],[-ks/mass,-cs/mass,ks/mass,cs/mass],
                  [0,0,0,1],[ks/MU,cs/MU,-(ks+KT)/MU,-cs/MU]], dtype=float)
    b = np.array([0,1/mass,0,-1/MU])
    e = np.array([0,0,0,KT/MU])
    return a,b,e

def discrete(mass=300., ks=KS, cs=CS, dt=DT):
    a,b,e = continuous(mass,ks,cs)
    aug=np.zeros((6,6)); aug[:4,:4]=a; aug[:4,4]=b; aug[:4,5]=e
    z=expm(aug*dt)
    return z[:4,:4],z[:4,4],z[:4,5]

def outputs(x,u,r,mass=300.,ks=KS,cs=CS):
    acc=(-ks*(x[0]-x[2])-cs*(x[1]-x[3])+u)/mass
    travel=x[0]-x[2]
    tire=KT*(x[2]-r)
    return np.array([acc,travel,tire])

class MPC:
    """Dense condensed QP with OSQP; input variable is v = u / 500 N."""
    def __init__(self,mass=300.,horizon=HORIZON,limit=500.):
        self.mass,self.n,self.limit=mass,horizon,limit
        self.ad,self.bd,self.ed=discrete(mass)
        a,b,_=continuous(mass)
        c=np.array([a[1],np.array([1,0,-1,0])/TRAVEL_SCALE,
                    np.array([0,0,KT,0])/LOAD_SCALE])
        d=np.array([b[1],0,0])*FORCE_SCALE
        er=np.array([0,0,-KT/LOAD_SCALE])
        px=np.eye(4); pu=np.zeros((4,horizon)); pr=np.zeros((4,horizon))
        gx=[]; gu=[]; gr=[]
        for i in range(horizon):
            ux=c@pu; ux[:,i]+=d
            rx=c@pr; rx[:,i]+=er
            gx.append(c@px); gu.append(ux); gr.append(rx)
            px=self.ad@px; pu=self.ad@pu; pr=self.ad@pr
            pu[:,i]+=self.bd*FORCE_SCALE; pr[:,i]+=self.ed
        w=np.tile(np.sqrt(WEIGHTS),horizon)
        self.gx=np.vstack(gx)*w[:,None]
        self.gu=np.vstack(gu)*w[:,None]
        self.gr=np.vstack(gr)*w[:,None]
        self.h=2*(self.gu.T@self.gu+EFFORT*np.eye(horizon))
        self.fx=2*self.gu.T@self.gx; self.fr=2*self.gu.T@self.gr
        self.solver=osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(np.triu(self.h)),q=np.zeros(horizon),
                          A=sparse.eye(horizon,format='csc'),
                          l=np.full(horizon,-limit/FORCE_SCALE),u=np.full(horizon,limit/FORCE_SCALE),
                          verbose=False,eps_abs=1e-7,eps_rel=1e-7,max_iter=20000,
                          polishing=False,adaptive_rho=True,check_termination=10)
        self.last=None
    def solve(self,x,road):
        f=self.fx@x+self.fr@road
        self.solver.update(q=f)
        result=self.solver.solve(raise_error=True)
        if result.info.status != 'solved':
            raise RuntimeError('Unaccepted QP status: '+result.info.status)
        v=result.x
        residual=np.max(np.abs(v-np.clip(v-(self.h@v+f),-self.limit/FORCE_SCALE,self.limit/FORCE_SCALE)))
        self.last={'kkt':float(residual),'primal':float(result.info.prim_res),
                   'dual':float(result.info.dual_res),'iterations':int(result.info.iter),
                   'raw_violation_N':float(max(0,np.max(np.abs(v))*FORCE_SCALE-self.limit))}
        return float(np.clip(v[0]*FORCE_SCALE,-self.limit,self.limit)),v.copy()

class MassRLS:
    """a_measured = (net_force / 300) theta + noise, theta = 300 / mass.

    Exact force regressor is an explicit simulation assumption. Update after
    actuation; the resulting mass is first used at the following control step.
    """
    def __init__(self,forget=.99):
        self.theta=1.; self.p=100.; self.forget=forget; self.updates=0
    @property
    def mass(self):
        return 300/self.theta
    def update(self,x,u,measured_acc):
        phi=(-KS*(x[0]-x[2])-CS*(x[1]-x[3])+u)/300
        if abs(phi) < .05:
            return self.mass
        gain=self.p*phi/(self.forget+phi*phi*self.p)
        self.theta=float(np.clip(self.theta+gain*(measured_acc-phi*self.theta),300/450,300/220))
        self.p=float(np.clip((self.p-gain*phi*self.p)/self.forget,1e-8,1e4))
        self.updates+=1
        return self.mass

def features(x,road,mass):
    return np.r_[x,road,mass]

class Policy:
    def __init__(self,path):
        z=np.load(path)
        self.mean=z['mean']; self.scale=z['scale']
        self.w=[z[f'w{i}'] for i in range(3)]; self.b=[z[f'b{i}'] for i in range(3)]
    def predict(self,xx):
        h=(xx-self.mean)/self.scale
        for w,b in zip(self.w[:-1],self.b[:-1]): h=np.maximum(h@w+b,0)
        return (h@self.w[-1]+self.b[-1]).reshape(-1)*FORCE_SCALE
    def action(self,x,road,mass):
        raw=float(self.predict(features(x,road,mass))[0])
        return float(np.clip(raw,-500,500)),raw

def road_profile(kind,seed,duration=6.,amplitude=None,frequency=None):
    rng=np.random.default_rng(seed)
    t=np.arange(int(duration/DT)+101)*DT
    ramp=np.minimum(t/.4,1.)**2
    if kind in ('sine','unseen_sine'):
        amp=amplitude if amplitude is not None else rng.uniform(.01,.025)
        freq=frequency if frequency is not None else rng.uniform(.8,3.)
        road=amp*np.sin(2*np.pi*freq*t+rng.uniform(0,2*np.pi))*ramp
    elif kind=='bump':
        height=amplitude if amplitude is not None else rng.uniform(.03,.05)
        width=rng.uniform(.22,.40); start=1.+rng.uniform(0,.2)
        phase=(t-start)/width
        road=np.where((phase>=0)&(phase<=1),height/2*(1-np.cos(2*np.pi*phase)),0.)
    elif kind=='multisine':
        freq=rng.uniform(.5,5.,size=8); phase=rng.uniform(0,2*np.pi,size=8)
        road=np.sum(np.sin(2*np.pi*t[:,None]*freq+phase)/np.sqrt(freq)[None,:],axis=1)
        road*= (amplitude or .009)/np.sqrt(np.mean(road**2)); road*=ramp
    elif kind=='flat': road=np.zeros_like(t)
    else: raise ValueError(kind)
    return t,road

def rollout(kind,seed,method='MPC',mass=300.,duration=6.,policy=None,horizon=HORIZON,
            limit=500.,amplitude=None,frequency=None,preview_noise=0.,preview_shift=0,
            mass_schedule=None,acc_noise=.05,ks=KS,cs=CS):
    t,road=road_profile(kind,seed,duration,amplitude,frequency)
    count=int(duration/DT)
    rng=np.random.default_rng(seed+80000)
    sensor=road+rng.normal(0,preview_noise,len(road))
    n=horizon if 'MPC' in method else HORIZON
    qps={}; plants={}; estimator=MassRLS()
    xs=[]; us=[]; raw=[]; yy=[]; masses=[]; estimates=[]; labels=[]; kkt=[]
    x=np.zeros(4)
    for k in range(count):
        true_mass=mass if mass_schedule is None else mass_schedule(t[k])
        if true_mass not in plants: plants[true_mass]=discrete(true_mass,ks,cs)
        m_est=estimator.mass if 'adaptive' in method else (300. if 'fixed' in method else true_mass)
        idx=np.clip(k+np.arange(n)+preview_shift,0,len(sensor)-1)
        preview=sensor[idx]
        if method=='MPC_no_preview': preview=np.full(n,sensor[k])
        if 'MPC' in method:
            # Fixed-mass evaluation only; identification experiments use the NN.
            if m_est not in qps: qps[m_est]=MPC(m_est,n,limit)
            u,_=qps[m_est].solve(x,preview); raw_u=u
            kkt.append(qps[m_est].last['kkt'])
        elif 'NN' in method:
            u,raw_u=policy.action(x,preview,m_est)
        elif method=='FFPD':
            # u = -ks*z_u - cs*v_u; exactly cancels base feedthrough in body equation when unsaturated.
            raw_u=-KS*x[2]-CS*x[3]; u=float(np.clip(raw_u,-limit,limit))
        elif method=='passive': u=raw_u=0.
        else: raise ValueError(method)
        y=outputs(x,u,road[k],true_mass,ks,cs)
        xs.append(x.copy()); us.append(u); raw.append(raw_u); yy.append(y)
        masses.append(true_mass); estimates.append(m_est)
        estimator.update(x,u,y[0]+rng.normal(0,acc_noise))
        ad,bd,ed=plants[true_mass]; x=ad@x+bd*u+ed*road[k]
    return dict(t=t[:count],road=road[:count],x=np.array(xs),u=np.array(us),raw=np.array(raw),
                y=np.array(yy),mass=np.array(masses),estimate=np.array(estimates),kkt=np.array(kkt))

def metrics(trace,limit=500.,start=0.):
    sl=trace['t']>=start; y=trace['y'][sl]; u=trace['u'][sl]; raw=trace['raw'][sl]
    fz=(trace['mass'][sl]+MU)*9.81-y[:,2]
    return dict(acc_rms=float(np.sqrt(np.mean(y[:,0]**2))),acc_peak=float(max(abs(y[:,0]))),
                travel_rms_mm=float(np.sqrt(np.mean(y[:,1]**2))*1000),
                travel_peak_mm=float(max(abs(y[:,1]))*1000),
                tire_rms_N=float(np.sqrt(np.mean(y[:,2]**2))),
                min_contact_ratio=float(np.min(fz/((trace['mass'][sl]+MU)*9.81))),
                force_rms_N=float(np.sqrt(np.mean(u**2))),saturation_pct=float(np.mean(abs(u)>=.99*limit)*100),
                violation_pct=float(np.mean(abs(u)>limit+1e-6)*100),
                raw_violation_pct=float(np.mean(abs(raw)>limit+1e-6)*100))
