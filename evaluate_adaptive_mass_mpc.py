"""Additional MPC-only mass adaptation study, using the frozen manuscript protocol.
Run: python evaluate_adaptive_mass_mpc.py
No tuning, NN inference, or changes to existing results. All 96 cases retained.
"""
import bootstrap
from bootstrap import ROOT
import core
from mass_controller import MassEstimator
from mass_optimization import WINDOWS, savecsv
from threadpoolctl import threadpool_limits
import numpy as np
import json, hashlib, time
from pathlib import Path
OUT=ROOT/'results/adaptive_mass_mpc';BASE=ROOT/'results/mass_optimization/mass_heads_final/heldout_traces'
METHODS=['MPC_fixed','MPC_adaptive','MPC_oracle']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def simulate(kind,seed,mass,method):
 t,road=core.road_profile(kind,seed,24);count=2400;rng=np.random.default_rng(seed+80000)
 rng.normal(0,0,len(road)) # same random stream as archived manuscript runs
 noise=rng.normal(0,.05,count);est=MassEstimator();x=np.zeros(4);qp=None
 plants={m:core.discrete(m) for m in [300.,mass]};rows=[];rebuilds=0
 for k in range(count):
  actual=mass if 4<=t[k]<20 else 300.
  model_mass=est.mass if method=='MPC_adaptive' else (300. if method=='MPC_fixed' else actual)
  if qp is None or qp.mass!=model_mass:qp=core.MPC(model_mass);rebuilds+=1
  u,_=qp.solve(x,road[k:k+30]);y=core.outputs(x,u,road[k],actual)
  rows.append((x.copy(),u,y,actual,model_mass,qp.last['kkt']))
  est.update(x,u,y[0]+noise[k]);a,b,e=plants[actual];x=a@x+b*u+e*road[k]
 return dict(t=t[:count],road=road[:count],x=np.array([r[0] for r in rows]),u=np.array([r[1] for r in rows]),raw=np.array([r[1] for r in rows]),y=np.array([r[2] for r in rows]),mass=np.array([r[3] for r in rows]),estimate=np.array([r[4] for r in rows]),kkt=np.array([r[5] for r in rows]),acc_noise=noise)
def validate(tr,kind,seed,mass,mode):
 t,road=core.road_profile(kind,seed,24);assert np.array_equal(tr['road'],road[:2400])
 expected=np.where((tr['t']>=4)&(tr['t']<20),mass,300.);assert np.array_equal(expected,tr['mass'])
 x=tr['x'];u=tr['u'];res=0.
 for m in np.unique(expected):
  ii=np.flatnonzero(expected[:-1]==m);a,b,e=core.discrete(m)
  res=max(res,float(np.max(abs(x[ii+1]-(x[ii]@a.T+u[ii,None]*b+tr['road'][ii,None]*e)))))
 acc=(-core.KS*(x[:,0]-x[:,2])-core.CS*(x[:,1]-x[:,3])+u)/expected
 assert res<1e-10 and np.max(abs(acc-tr['y'][:,0]))<1e-10 and np.max(abs(u))<=500+1e-7
 if mode=='MPC_adaptive':
  est=MassEstimator()
  for k in range(len(u)):
   assert abs(est.mass-tr['estimate'][k])<1e-9
   est.update(x[k],u[k],tr['y'][k,0]+tr['acc_noise'][k])
 else:assert np.array_equal(tr['estimate'],np.full(len(u),300.) if mode=='MPC_fixed' else expected)
 return res

def main():
 OUT.mkdir(exist_ok=True);td=OUT/'traces';td.mkdir(exist_ok=True)
 protocol=dict(duration_s=24,change_times_s=[4,20],masses_kg=[200,240,360,420],roads={'sine':list(range(16000,16012)),'multisine':list(range(16100,16112))},methods=METHODS,windows_s=WINDOWS,acceleration_noise_std=.05,rls={'forget':.97,'gate':.02,'mass_bounds':[180,480],'initial_mass':300,'initial_covariance':100},controller='Identical core.MPC objective, 30-step preview, 10 ms sampling, ±500 N; rebuild with each changed prior-sample estimate, no rounding or smoothing.',causality='Measure acceleration after current action, update mass for next step. Only oracle receives current true mass; no future mass schedule in prediction.',primary='Group reduction = 100*(mean fixed RMS - mean adaptive RMS)/mean fixed RMS over 4–20 s; also individual-road counts and early/late windows.',sources={n:digest(ROOT/n) for n in ['core.py','mass_controller.py','evaluate_adaptive_mass_mpc.py']},selection='No tuning or selecting by adaptive improvement. Plot upper-median fixed MPC loaded RMS at 200 kg, sinusoidal roads.')
 pf=OUT/'protocol.json'
 if pf.exists():assert json.loads(pf.read_text())==json.loads(json.dumps(protocol))
 else:pf.write_text(json.dumps(protocol,indent=2))
 rows=[];maxres=0.;start=time.time();reused=0
 for kind,seeds in protocol['roads'].items():
  for seed in seeds:
   for mass in protocol['masses_kg']:
    for mode in METHODS:
     name=f'{kind}_{seed}_{mass}_{mode}.npz';p=td/name
     if mode!='MPC_adaptive' and (BASE/name).exists():p=BASE/name;reused+=1
     if p.exists():tr=dict(np.load(p))
     else:tr=simulate(kind,seed,mass,mode);np.savez_compressed(p,**tr)
     maxres=max(maxres,validate(tr,kind,seed,mass,mode))
     for win,(a,b) in WINDOWS.items():
      mask=(tr['t']>=a)&(tr['t']<b);sub={k:v[mask] for k,v in tr.items() if len(v)==2400}
      error=abs(tr['estimate']-tr['mass'])/tr['mass']*100;tail=(tr['t']>=18)&(tr['t']<20)
      rows.append(dict(kind=kind,seed=seed,mass=mass,mode=mode,window=win,tail_error_pct=float(error[tail].mean()),**core.metrics(sub)))
    print(kind,seed,mass,'elapsed',round(time.time()-start),flush=True)
 savecsv(OUT/'per_road.csv',rows);summary=[]
 for kind in protocol['roads']:
  for mass in protocol['masses_kg']:
   for win in WINDOWS:
    rr=[r for r in rows if r['kind']==kind and r['mass']==mass and r['window']==win];means={m:{k:float(np.mean([r[k] for r in rr if r['mode']==m])) for k in ['acc_rms','acc_peak','travel_rms_mm','tire_rms_N','force_rms_N','tail_error_pct']} for m in METHODS}
    f={r['seed']:r['acc_rms'] for r in rr if r['mode']=='MPC_fixed'};ad={r['seed']:r['acc_rms'] for r in rr if r['mode']=='MPC_adaptive'}
    summary.append(dict(kind=kind,mass=mass,window=win,means=means,adaptive_gain_pct=100*(1-means['MPC_adaptive']['acc_rms']/means['MPC_fixed']['acc_rms']),oracle_gain_pct=100*(1-means['MPC_oracle']['acc_rms']/means['MPC_fixed']['acc_rms']),improved_roads=sum(ad[s]<f[s] for s in f),roads=len(f)))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'validation.json').write_text(json.dumps(dict(cases=96,trajectories=288,reused_reference_trajectories=reused,max_transition_residual=maxres,all_force_bounds_pass=True,all_causal_estimator_replays_pass=True,minimum_contact_ratio=min(r['min_contact_ratio'] for r in rows)),indent=2))
 plots(rows,summary)
 for r in summary:
  if r['window']=='loaded':print('RESULT',r['kind'],r['mass'],round(r['adaptive_gain_pct'],4),round(r['oracle_gain_pct'],4),r['improved_roads'],flush=True)
def plots(rows,summary):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.25,'axes.spines.top':True,'axes.spines.right':True,'pdf.fonttype':42})
 pool=sorted([r for r in rows if r['kind']=='sine' and r['mass']==200 and r['window']=='loaded' and r['mode']=='MPC_fixed'],key=lambda r:r['acc_rms']);seed=pool[len(pool)//2]['seed'];trs={}
 for mode in METHODS:
  name=f'sine_{seed}_200_{mode}.npz';p=OUT/'traces'/name
  if not p.exists():p=BASE/name
  trs[mode]=dict(np.load(p))
 fig,axs=plt.subplots(2,1,figsize=(3.7,4.9));t=trs['MPC_adaptive']['t'];axs[0].plot(t,trs['MPC_adaptive']['mass'],'k--',label='True mass');axs[0].plot(t,trs['MPC_adaptive']['estimate'],'r-',lw=.8,label='Online estimate');axs[0].set(xlabel='Time [s]',ylabel='Sprung mass [kg]',title='(a) MPC mass tracking');axs[0].legend(fontsize=6)
 sl=(t>=4)&(t<20)
 for mode,c,ls,label in zip(METHODS,['blue','red','#009FA8'],['-','--','-.'],['Fixed MPC','Adaptive MPC','True mass MPC']):
  a=trs[mode]['y'][sl,0];axs[1].plot(10*(t[sl]-4),np.sqrt(np.cumsum(a*a)/np.arange(1,len(a)+1)),color=c,ls=ls,label=label)
 axs[1].set(xlabel='Distance after mass change [m]',ylabel='Acceleration RMS [m/s²]',title='(b) Cumulative response');axs[1].legend(fontsize=6);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(OUT/f'mpc_mass_adaptation.{ext}',dpi=300)
 plt.close(fig)
 lines=['# MPC online mass identification comparison','','Same frozen mass study; no NN or controller retuning. Positive reduction means lower acceleration RMS.','','| Road | Mass (kg) | Fixed RMS | Adaptive RMS | True mass RMS | Adaptive reduction (%) | True mass reduction (%) | Improved roads |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in summary:
  if r['window']=='loaded':
   v=r['means'];lines.append(f"| {r['kind']} | {r['mass']} | {v['MPC_fixed']['acc_rms']:.5f} | {v['MPC_adaptive']['acc_rms']:.5f} | {v['MPC_oracle']['acc_rms']:.5f} | {r['adaptive_gain_pct']:.2f} | {r['oracle_gain_pct']:.2f} | {r['improved_roads']}/12 |")
 lines+=['','RMS unit: m/s². Primary window: 4–20 s (160 m at 10 m/s). All cases retained. Early 4–8 s, later 8–20 s, full and recovery windows are in summary.json and per_road.csv.',f'Figure case: sine road {seed}, 200 kg; chosen by upper-median fixed MPC RMS, not adaptive gain.','Fixed/oracle traces reused from ../mass_optimization/mass_heads_final/heldout_traces. Newly simulated adaptive traces are under traces/.','The RLS tuning is inherited from the NN study. Results assess transferring that identifier to MPC; they do not establish the best achievable MPC adaptation.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
