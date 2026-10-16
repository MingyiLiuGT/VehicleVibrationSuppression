"""Exploratory operating-region study with a frozen policy and causal RLS.

Development scans identify physical conditions; a separately frozen validation
protocol tests them on fresh roads. No trajectory is removed for its outcome.
"""
import bootstrap
from bootstrap import ROOT
import core
from mass_controller import MassConditionedPolicy, MassEstimator
from threadpoolctl import threadpool_limits
import numpy as np
import csv, json, hashlib, argparse

OUT=ROOT/'results/mass_regime_study'
OUT.mkdir(exist_ok=True)
MODES=['NN_fixed','NN_adaptive','NN_oracle']
N=2400

def write_json(path,value):
 path.write_text(json.dumps(value,indent=2)+'\n')

def freeze(path,value):
 if path.exists(): assert json.loads(path.read_text())==value
 else: write_json(path,value)

def batch(cases,save_traces=False):
 """Vectorized equivalent of core.rollout; same noise stream and equations."""
 policy=MassConditionedPolicy();count=len(cases)
 roads=[];noise=[]
 for c in cases:
  _,r=core.road_profile(c['kind'],c['seed'],24,amplitude=c.get('amplitude'),frequency=c.get('frequency'))
  rng=np.random.default_rng(c['seed']+80000);rng.normal(0,0,len(r))
  roads.append(r);noise.append(rng.normal(0,.05,N))
 roads=np.array(roads);noise=np.array(noise)
 changed=np.array([c['mass'] for c in cases]);adaptive=np.array([c['mode']=='NN_adaptive' for c in cases]);fixed=np.array([c['mode']=='NN_fixed' for c in cases])
 plants={m:core.discrete(m) for m in set(changed)|{300.}}
 aa=np.array([plants[m][0] for m in changed]);bb=np.array([plants[m][1] for m in changed]);ee=np.array([plants[m][2] for m in changed])
 x=np.zeros((count,4));theta=np.ones(count);cov=np.full(count,100.)
 hist={k:np.zeros((count,N)+(shape,)) if shape else np.zeros((count,N)) for k,shape in [('x',4),('y',3),('u',0),('raw',0),('estimate',0),('mass',0)]}
 ii=np.arange(count)
 for k in range(N):
  active=400<=k<2000;mass=changed if active else np.full(count,300.)
  estimate=np.where(fixed,300.,np.where(adaptive,300/theta,mass))
  heads=policy.heads(np.column_stack((x,roads[:,k:k+30])))
  mm=np.clip(estimate,policy.grid[0],policy.grid[-1]);hi=np.clip(np.searchsorted(policy.grid,mm,side='right'),1,len(policy.grid)-1);lo=hi-1
  alpha=(mm-policy.grid[lo])/(policy.grid[hi]-policy.grid[lo]);raw=(1-alpha)*heads[ii,lo]+alpha*heads[ii,hi];u=np.clip(raw,-500,500)
  force=-core.KS*(x[:,0]-x[:,2])-core.CS*(x[:,1]-x[:,3])+u
  y=np.column_stack((force/mass,x[:,0]-x[:,2],core.KT*(x[:,2]-roads[:,k])))
  for key,value in [('x',x),('y',y),('u',u),('raw',raw),('mass',mass),('estimate',estimate)]:hist[key][:,k]=value
  phi=force/300;gate=abs(phi)>=.02;gain=cov*phi/(.97+phi*phi*cov)
  theta=np.where(gate,np.clip(theta+gain*(y[:,0]+noise[:,k]-phi*theta),300/480,300/180),theta)
  cov=np.where(gate,np.clip((cov-gain*phi*cov)/.97,1e-8,1e4),cov)
  if active:x=np.einsum('bij,bj->bi',aa,x)+bb*u[:,None]+ee*roads[:,k,None]
  else:
   a,b,e=plants[300.];x=x@a.T+u[:,None]*b+roads[:,k,None]*e
 rows=[]
 for i,c in enumerate(cases):
  tr={key:value[i] for key,value in hist.items()};tr['road']=roads[i,:N];tr['t']=np.arange(N)*.01
  if save_traces:
   dest=OUT/save_traces;dest.mkdir(exist_ok=True);np.savez_compressed(dest/(c['case_id']+'_'+c['mode']+'.npz'),**tr)
  row=dict(c)
  for name,a,b in [('loaded',400,2000),('early',400,800),('settled',800,2000),('terminal',1800,2000)]:
   row[name+'_rms']=float(np.sqrt(np.mean(tr['y'][a:b,0]**2)))
  row.update(core.metrics({key:value[400:2000] for key,value in tr.items()}))
  row['terminal_error_pct']=float(np.mean(abs(tr['estimate'][1800:2000]-tr['mass'][1800:2000])/tr['mass'][1800:2000])*100)
  rows.append(row)
 return rows,hist

def evaluate(cases,label,traces=False):
 freeze(OUT/(label+'_cases.json'),cases);rows=[]
 for start in range(0,len(cases),96):
  rr,_=batch(cases[start:start+96],label+'_traces' if traces else False);rows+=rr
  print(label,start+len(rr),'/',len(cases),flush=True)
 with (OUT/(label+'.csv')).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 groups=[]
 for group in dict.fromkeys(c['group'] for c in cases):
  rr=[r for r in rows if r['group']==group];means={mode:{key:float(np.mean([r[key] for r in rr if r['mode']==mode])) for key in ['loaded_rms','early_rms','settled_rms','terminal_error_pct','force_rms_N','travel_rms_mm','tire_rms_N']} for mode in MODES}
  bymode={mode:{r['seed']:r['loaded_rms'] for r in rr if r['mode']==mode} for mode in MODES}
  gains={mode:100*(1-means[mode]['loaded_rms']/means['NN_fixed']['loaded_rms']) for mode in MODES}
  groups.append(dict(group=group,n=len(bymode['NN_fixed']),means=means,gains_pct=gains,improved=sum(bymode['NN_adaptive'][s]<bymode['NN_fixed'][s] for s in bymode['NN_fixed']),min_contact=min(r['min_contact_ratio'] for r in rr)))
 write_json(OUT/(label+'_summary.json'),groups)
 return groups

def verify():
 core.MassRLS=MassEstimator;policy=MassConditionedPolicy();cases=[]
 for mass in [200.,450.]:
  for mode in MODES:cases.append(dict(case_id=f'check_{mass}',group='check',kind='sine',seed=19000,mass=mass,mode=mode,amplitude=.015,frequency=1.2))
 _,h=batch(cases);err={key:0. for key in h}
 for i,c in enumerate(cases):
  tr=core.rollout(c['kind'],c['seed'],c['mode'],duration=24,policy=policy,mass_schedule=lambda t:c['mass'] if 4<=t<20 else 300.,amplitude=c['amplitude'],frequency=c['frequency'])
  for key in h:err[key]=max(err[key],float(np.max(abs(h[key][i]-tr[key]))))
 assert max(err.values())<1e-6,err
 write_json(OUT/'batch_verification.json',err);print(err,flush=True)

def mass_scan():
 cases=[]
 for kind,offset in [('sine',19000),('multisine',19100)]:
  for mass in [200,225,240,250,270,300,330,350,360,375,400,420,450]:
   for seed in range(offset,offset+6):
    for mode in MODES:cases.append(dict(case_id=f'{kind}_{mass}_{seed}',group=f'{kind}_{mass}',kind=kind,seed=seed,mass=mass,mode=mode,amplitude=None,frequency=None))
 return evaluate(cases,'mass_development')

def excitation_scan():
 cases=[]
 for mass in [200,240,360,420,450]:
  for amplitude in [.005,.01,.015,.02,.025]:
   for frequency in [.4,.6,.8,1.,1.2,1.4,1.6,1.8,2.,2.5,3.,4.,5.]:
    group=f'm{mass}_a{amplitude}_f{frequency}'
    for seed in range(19200,19204):
     for mode in MODES:cases.append(dict(case_id=f'{group}_{seed}',group=group,kind='sine',seed=seed,mass=mass,mode=mode,amplitude=amplitude,frequency=frequency))
 return evaluate(cases,'excitation_development')

def neighborhood_scan():
 cases=[]
 for masses,amps,freqs in [([200,220,240],[.018,.019,.02,.021,.022],[1.1,1.2,1.3,1.4,1.5,1.6]),([360,400,420,450],[.023,.024,.025,.026,.027],[.5,.6,.7,.8,.9,1.])]:
  for mass in masses:
   for amplitude in amps:
    for frequency in freqs:
     group=f'm{mass}_a{amplitude}_f{frequency}'
     for seed in range(19300,19304):
      for mode in MODES:cases.append(dict(case_id=f'{group}_{seed}',group=group,kind='sine',seed=seed,mass=mass,mode=mode,amplitude=amplitude,frequency=frequency))
 return evaluate(cases,'neighborhood_development')

def validation_cases():
 protocol=dict(n_roads_per_direction=24,primary_masses_kg=[200,420],neighboring_masses_kg=[220,240,360,400],decrease=dict(masses=[200,220,240],amplitude_m=[.019,.021],frequency_hz=[1.3,1.5],road_ids=list(range(19400,19424))),increase=dict(masses=[360,400,420],amplitude_m=[.024,.026],frequency_hz=[.6,.8],road_ids=list(range(19500,19524))),selection='Regions fixed after development scans; primary masses chosen to retain the previous 200/420 kg comparison. Every fresh validation road and every mode retained, including failures. Region-specific interpretation only.',success='Positive group mean NN adaptive and oracle gains for both primary directions, at least 20 of 24 roads improving, no force violation or tire contact loss. Report absolute RMS and neighboring mass results.',representative='Upper median fixed-controller loaded RMS, without looking at adaptive gain. Figures use all-road aggregate curves rather than selecting individual improvements.')
 freeze(OUT/'validation_protocol.json',protocol);cases=[]
 for direction in ['decrease','increase']:
  s=protocol[direction]
  for mass in s['masses']:
   for seed in s['road_ids']:
    rng=np.random.default_rng(seed+50000);amp=float(rng.uniform(*s['amplitude_m']));freq=float(rng.uniform(*s['frequency_hz']))
    group=f'{direction}_{mass}'
    for mode in MODES:cases.append(dict(case_id=f'{group}_{seed}',group=group,kind='sine',seed=seed,mass=mass,mode=mode,amplitude=amp,frequency=freq))
 return cases

def validate_regions():
 return evaluate(validation_cases(),'validation',traces=True)

def validate_mpc():
 core.MassRLS=MassEstimator;cases=validation_cases();rows=[]
 for c in cases:
  if c['mode']!='NN_fixed':continue
  for mode in ['MPC_fixed','MPC_oracle']:
   trace=core.rollout('sine',c['seed'],mode,duration=24,mass_schedule=lambda t:c['mass'] if 4<=t<20 else 300.,amplitude=c['amplitude'],frequency=c['frequency'])
   dest=OUT/'validation_traces';dest.mkdir(exist_ok=True);np.savez_compressed(dest/(c['case_id']+'_'+mode+'.npz'),**trace)
   row=dict(c,mode=mode);row.update(core.metrics({key:value[400:2000] for key,value in trace.items()}));rows.append(row)
  if len(rows)%24==0:print('MPC validation',len(rows),'/ 288',flush=True)
 with (OUT/'validation_mpc.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for group in dict.fromkeys(c['group'] for c in cases):
  means={mode:float(np.mean([r['acc_rms'] for r in rows if r['group']==group and r['mode']==mode])) for mode in ['MPC_fixed','MPC_oracle']}
  summary.append(dict(group=group,means=means,gain_pct=100*(1-means['MPC_oracle']/means['MPC_fixed'])))
 write_json(OUT/'validation_mpc_summary.json',summary);print(summary,flush=True)

def audit():
 cases=json.loads((OUT/'validation_cases.json').read_text());rows=list(csv.DictReader((OUT/'validation.csv').open()))
 errors=dict(dynamics=0.,force_balance=0.,policy_replay=0.,estimator_replay=0.,rms=0.)
 max_force=0.;min_contact=1.;policy=MassConditionedPolicy();count=0
 for c in cases:
  tr=np.load(OUT/'validation_traces'/(c['case_id']+'_'+c['mode']+'.npz'));count+=1
  x,u,y,m=tr['x'],tr['u'],tr['y'],tr['mass']
  for mass in np.unique(m):
   ix=np.flatnonzero(m[:-1]==mass);a,b,e=core.discrete(mass)
   errors['dynamics']=max(errors['dynamics'],float(np.max(abs(x[ix+1]-(x[ix]@a.T+u[ix,None]*b+tr['road'][ix,None]*e)))))
  expected=(-core.KS*(x[:,0]-x[:,2])-core.CS*(x[:,1]-x[:,3])+u)/m
  errors['force_balance']=max(errors['force_balance'],float(np.max(abs(y[:,0]-expected))))
  _,road=core.road_profile('sine',c['seed'],24,c['amplitude'],c['frequency']);ix=np.arange(0,N,19)
  xx=np.array([np.r_[x[k],road[k:k+30],tr['estimate'][k]] for k in ix]);expected=np.clip(policy.predict(xx),-500,500)
  errors['policy_replay']=max(errors['policy_replay'],float(np.max(abs(expected-u[ix]))))
  if c['mode']=='NN_adaptive':
   rng=np.random.default_rng(c['seed']+80000);rng.normal(0,0,len(road));noise=rng.normal(0,.05,N);est=MassEstimator()
   for k in range(N):
    errors['estimator_replay']=max(errors['estimator_replay'],abs(est.mass-tr['estimate'][k]));est.update(x[k],u[k],y[k,0]+noise[k])
  r=next(r for r in rows if r['case_id']==c['case_id'] and r['mode']==c['mode'])
  errors['rms']=max(errors['rms'],abs(float(np.sqrt(np.mean(y[400:2000,0]**2)))-float(r['loaded_rms'])))
  max_force=max(max_force,float(max(abs(u))));min_contact=min(min_contact,float(np.min(1-y[:,2]/((m+40)*9.81))))
 mpc_rows=list(csv.DictReader((OUT/'validation_mpc.csv').open()))
 mpc_lookup={r['case_id']+'_'+r['mode']:r for r in mpc_rows}
 for path in sorted((OUT/'validation_traces').glob('*MPC*.npz')):
  tr=np.load(path);count+=1;x,u,y,m=tr['x'],tr['u'],tr['y'],tr['mass']
  for mass in np.unique(m):
   ix=np.flatnonzero(m[:-1]==mass);a,b,e=core.discrete(mass)
   errors['dynamics']=max(errors['dynamics'],float(np.max(abs(x[ix+1]-(x[ix]@a.T+u[ix,None]*b+tr['road'][ix,None]*e)))))
  errors['force_balance']=max(errors['force_balance'],float(np.max(abs(y[:,0]-(-core.KS*(x[:,0]-x[:,2])-core.CS*(x[:,1]-x[:,3])+u)/m))))
  assert max(tr['kkt'])<1e-5
  errors['rms']=max(errors['rms'],abs(float(np.sqrt(np.mean(y[400:2000,0]**2)))-float(mpc_lookup[path.stem]['acc_rms'])))
  max_force=max(max_force,float(max(abs(u))));min_contact=min(min_contact,float(np.min(1-y[:,2]/((m+40)*9.81))))
 assert count==720 and max(errors.values())<1e-7 and max_force<=500 and min_contact>0
 summary=json.loads((OUT/'validation_summary.json').read_text());assert all(r['n']==24 and r['improved']==24 and r['gains_pct']['NN_adaptive']>0 for r in summary)
 report=dict(trajectories=count,max_errors=errors,max_force_N=max_force,min_contact_ratio=min_contact,all_validation_roads_retained=True,primary_success=True)
 write_json(OUT/'validation_audit.json',report);print(report,flush=True)

def figures():
 from analyze_results import plt,finish,panel
 dest=OUT/'figures';dest.mkdir(exist_ok=True)
 cases=json.loads((OUT/'validation_cases.json').read_text());t=np.arange(N)*.01
 fig1,axs1=plt.subplots(2,1,figsize=(3.7,4.9));fig2,axs2=plt.subplots(2,1,figsize=(3.7,4.9));stats={}
 for i,(group,mass) in enumerate([('decrease_200',200),('increase_420',420)]):
  rr=[c for c in cases if c['group']==group];traces={mode:[np.load(OUT/'validation_traces'/(c['case_id']+'_'+mode+'.npz')) for c in rr if c['mode']==mode] for mode in MODES}
  estimates=np.array([tr['estimate'] for tr in traces['NN_adaptive']]);mu=estimates.mean(0);sd=estimates.std(0,ddof=1);truth=traces['NN_adaptive'][0]['mass']
  axs1[i].fill_between(t,mu-sd,mu+sd,color='#FF0000',alpha=.12,lw=0,label='±1 SD')
  axs1[i].plot(t,truth,'k--',lw=1.05,label='True mass');axs1[i].plot(t,mu,'r-',lw=1.1,label='Mean estimate')
  axs1[i].set(xlim=(0,24),xticks=np.arange(0,25,4),xlabel='Time [s]',ylabel='Sprung mass [kg]')
  lo,hi=axs1[i].get_ylim();axs1[i].set_ylim(lo,hi+.25*(hi-lo));axs1[i].legend(fontsize=6,loc='upper right',ncol=1)
  for change in [4,20]:axs1[i].axvline(change,color='gray',ls=':',lw=.7)
  final={}
  for mode,color,style,label in zip(MODES,['#0000FF','#FF0000','#009FA8'],['-','--','-.'],['Without identification','With identification','True-mass input']):
   acc=np.array([tr['y'][400:2000,0] for tr in traces[mode]])
   curve=np.sqrt(np.cumsum(acc*acc,axis=1)/np.arange(1,1601)).mean(0)
   axs2[i].plot(np.arange(1600)*.1,curve,color=color,ls=style,lw=1.1,label=label);final[mode]=float(curve[-1])
  axs2[i].set(xlim=(0,160),xlabel='Distance after mass change [m]',ylabel='Cumulative acceleration\nRMS [m/s²]');axs2[i].legend(fontsize=6,loc='lower right')
  if i==0:axs2[i].set_ylim(0,.06)
  for ax in [axs1[i],axs2[i]]:panel(ax,'ab'[i],f'Mass change: 300 → {mass} kg')
  error=np.abs(estimates-truth)/truth*100
  stats[group]=dict(final_curve_rms=final,early_mass_error_pct=float(error[:,400:800].mean()),terminal_mass_error_pct=float(error[:,1800:2000].mean()))
 for fig,name in [(fig1,'mass_tracking'),(fig2,'mass_response')]:
  finish(fig)
  for ext in ['png','pdf','svg']:fig.savefig(dest/(name+'.'+ext),dpi=600,bbox_inches=None)
  plt.close(fig)
 write_json(OUT/'figure_statistics.json',stats)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['verify','mass','excitation','neighborhood','validation','mpc','audit','figures']);args=ap.parse_args()
 freeze(OUT/'protocol.json',dict(policy_sha256=hashlib.sha256((ROOT/'results/mass_optimization/mass_heads_final/policy.npz').read_bytes()).hexdigest(),mass_range_kg=[200,450],reference_mass_kg=300,duration_s=24,changed_interval_s=[4,20],noise_m_s2=.05,rls=dict(forget=.97,gate=.02,bounds=[180,480]),purpose='Exploratory operating region selection followed by independent validation; retain every exploratory result. No policy or fixed baseline changes.',primary='Ratio of group mean RMS over the entire 4-20 s interval; no outcome-dependent time window or road removal.'))
 with threadpool_limits(limits=1):
  if args.stage=='verify':verify()
  elif args.stage=='mass':
   for r in mass_scan():print(r['group'],r['gains_pct'],r['improved'],flush=True)
  elif args.stage=='excitation':excitation_scan()
  elif args.stage=='neighborhood':neighborhood_scan()
  elif args.stage=='validation':
   for r in validate_regions():print(r['group'],r['gains_pct'],r['improved'],flush=True)
  elif args.stage=='mpc':validate_mpc()
  elif args.stage=='audit':audit()
  else:figures()
