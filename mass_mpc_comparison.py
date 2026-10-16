"""Matched fixed/true mass MPC controls supplement the frozen NN mass study.

No training or controller tuning. Reuse all 60 existing roads and seven masses;
independent experimental unit remains the road. Same-state sensitivity queries
use every tenth state on the existing nominal NN trajectory, including t=0.
"""
import bootstrap
from bootstrap import ROOT
import core
import numpy as np
import csv,json,hashlib
from pathlib import Path
from mass_sensitivity_study import DELTAS,interval
OUT=ROOT/'results/mass_mpc_comparison';OUT.mkdir(exist_ok=True)
BASE=ROOT/'results/mass_sensitivity'
def savecsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def met(tr):
 r=core.metrics(tr)
 scaled=tr['y']/np.array([1.,core.TRAVEL_SCALE,core.LOAD_SCALE])
 r['stage_cost_mean']=float(np.mean((scaled**2)@core.WEIGHTS+core.EFFORT*(tr['u']/500)**2))
 return r

def main():
 protocol=dict(status='fixed before additional MPC evaluation',masses=[300*(1+d) for d in DELTAS],
  seeds={k:list(range(b,b+20)) for k,b in [('multisine',11000),('sine',11100),('bump',11200)]},
  duration_s=6,methods=['MPC_fixed','MPC_oracle'],reuse='Archived NN_fixed, NN_oracle and passive traces; no retraining.',
  policy_sha256=hashlib.sha256((ROOT/'results/models/policy.npz').read_bytes()).hexdigest(),
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  same_state='Every tenth nominal NN state; identical exact 30-step preview; evaluate seven model/input masses. 60 states per road.',
  switching='Reuse 8 mass-switching roads 6000–6007, 300→420→300 kg at 4 and 8 s, 12 s. MPC oracle knows current mass, not future switch.',
  gain='100*(fixed RMS - true-input RMS)/fixed RMS; paired by road. Positive means value from mass information.',
  cost='Full-run mean of normalized acceleration, travel, tire force and effort stage costs, not a finite-horizon value function.',
  inference='Descriptive 10000-resample percentile road bootstrap; repeated masses are not independent roads.')
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));td=OUT/'traces';td.mkdir(exist_ok=True)
 policy=core.Policy(ROOT/'results/models/policy.npz');rows=[];actions=[]
 maxres=0.;maxacc=0.;maxkkt=0.
 def verify(tr):
  nonlocal maxres,maxacc,maxkkt
  for m in np.unique(tr['mass']):
   idx=np.flatnonzero(tr['mass'][:-1]==m);a,b,e=core.discrete(m)
   residual=tr['x'][idx]@a.T+tr['u'][idx,None]*b+tr['road'][idx,None]*e-tr['x'][idx+1]
   maxres=max(maxres,float(np.max(abs(residual))))
  acc=(-core.KS*(tr['x'][:,0]-tr['x'][:,2])-core.CS*(tr['x'][:,1]-tr['x'][:,3])+tr['u'])/tr['mass']
  maxacc=max(maxacc,float(np.max(abs(acc-tr['y'][:,0]))));assert max(abs(tr['u']))<=500+1e-8
  if len(tr.get('kkt',[])):maxkkt=max(maxkkt,float(max(tr['kkt'])))
 for kind,seeds in protocol['seeds'].items():
  for seed in seeds:
   nominal=np.load(BASE/f'traces/{kind}_{seed}_m300_NN_fixed.npz');_,road=core.road_profile(kind,seed,6)
   sample=np.arange(0,600,10);queries={}
   for delta in DELTAS:
    mass=300*(1+delta)
    for method in ['NN_fixed','NN_oracle','passive','MPC_fixed','MPC_oracle']:
     if method.startswith('MPC'):
      tr=core.rollout(kind,seed,method,mass=mass,duration=6)
      np.savez_compressed(td/f'{kind}_{seed}_m{mass:.0f}_{method}.npz',**tr)
     else:tr=np.load(BASE/f'traces/{kind}_{seed}_m{mass:.0f}_{method}.npz')
     verify(tr);rows.append(dict(study='constant',kind=kind,seed=seed,mass=mass,delta=delta,method=method,**met(tr)))
    qp=core.MPC(mass)
    queries[delta]={m:[] for m in ['NN','MPC']}
    for i in sample:
     x=nominal['x'][i];pr=road[i:i+30]
     queries[delta]['NN'].append(policy.action(x,pr,mass)[0]);queries[delta]['MPC'].append(qp.solve(x,pr)[0])
   for delta in DELTAS:
    for method in ['NN','MPC']:
     a=np.array(queries[delta][method]);nom=np.array(queries[0.][method]);teacher=np.array(queries[delta]['MPC'])
     actions.append(dict(kind=kind,seed=seed,delta=delta,method=method,samples=len(a),action_difference_rms_N=float(np.sqrt(np.mean((a-nom)**2))),max_difference_N=float(max(abs(a-nom))),teacher_rmse_N=float(np.sqrt(np.mean((a-teacher)**2)))))
   print('mass comparison',kind,seed,flush=True)
 schedule=lambda t:300. if t<4 or t>=8 else 420.
 for seed in range(6000,6008):
  for method in ['NN_fixed','NN_oracle','MPC_fixed','MPC_oracle']:
   if method.startswith('NN'):tr=np.load(ROOT/f'results/traces/id_0.05_{seed}_{method}.npz')
   else:
    tr=core.rollout('multisine',seed,method,duration=12,mass_schedule=schedule)
    np.savez_compressed(td/f'switch_{seed}_{method}.npz',**tr)
   verify(tr);rows.append(dict(study='switching',kind='multisine',seed=seed,mass=420.,delta=.4,method=method,**met(tr)))
 savecsv('metrics.csv',rows);savecsv('same_state_actions.csv',actions)
 groups={}
 for kind in protocol['seeds']:
  groups[kind]={}
  for delta in DELTAS:
   rr=[r for r in rows if r['study']=='constant' and r['kind']==kind and r['delta']==delta];groups[kind][str(delta)]={}
   for model in ['NN','MPC']:
    f={r['seed']:r for r in rr if r['method']==model+'_fixed'};o={r['seed']:r for r in rr if r['method']==model+'_oracle'}
    a=[r for r in actions if r['kind']==kind and r['delta']==delta and r['method']==model]
    groups[kind][str(delta)][model]=dict(gain_pct=interval([100*(1-o[k]['acc_rms']/f[k]['acc_rms']) for k in f]),
     fixed_mean_rms=float(np.mean([r['acc_rms'] for r in f.values()])),oracle_mean_rms=float(np.mean([r['acc_rms'] for r in o.values()])),
     cost_gain_pct=interval([100*(1-o[k]['stage_cost_mean']/f[k]['stage_cost_mean']) for k in f]),
     same_state_rms_N=float(np.mean([r['action_difference_rms_N'] for r in a])),
     same_state_max_N=max(r['max_difference_N'] for r in a),teacher_rmse_N=float(np.mean([r['teacher_rmse_N'] for r in a])))
 switching={}
 for model in ['NN','MPC']:
  f={r['seed']:r for r in rows if r['study']=='switching' and r['method']==model+'_fixed'};o={r['seed']:r for r in rows if r['study']=='switching' and r['method']==model+'_oracle'}
  switching[model]=dict(gain_pct=interval([100*(1-o[k]['acc_rms']/f[k]['acc_rms']) for k in f]),fixed_mean_rms=float(np.mean([r['acc_rms'] for r in f.values()])),oracle_mean_rms=float(np.mean([r['acc_rms'] for r in o.values()])))
 summary=dict(groups=groups,switching=switching,validation=dict(new_mpc_rollouts=856,reused_nn_passive_rollouts=1276,max_transition_residual=maxres,max_acceleration_residual=maxacc,max_kkt=maxkkt))
 assert maxres<1e-10 and maxacc<1e-10
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',json.dumps(switching),flush=True)
if __name__=='__main__':main()
