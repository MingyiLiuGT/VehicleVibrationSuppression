"""Audit all mass-sensitivity traces and report favorable and adverse results."""
import bootstrap
from bootstrap import ROOT
import core
from analyze_results import plt,finish,panel
import numpy as np,json,csv
from mass_sensitivity_study import interval
OUT=ROOT/'results/mass_sensitivity';s=json.loads((OUT/'summary.json').read_text());rows=list(csv.DictReader((OUT/'metrics.csv').open()))
for r in rows:
 for k in r:
  if k not in ['kind','method']:r[k]=float(r[k])
lookup={(r['kind'],r['seed'],r['delta'],r['method']):r for r in rows}
max_res=0.;max_acc=0.;count=0;maxforce=0.;mincontact=1e9
plants={m:core.discrete(m) for m in [240.,270.,285.,300.,315.,330.,360.]}
for path in (OUT/'traces').glob('*.npz'):
 with np.load(path) as z:
  count+=1;x=z['x'];u=z['u'];road=z['road'];m=float(z['mass'][0]);a,b,e=plants[m]
  pred=x[:-1]@a.T+u[:-1,None]*b+road[:-1,None]*e
  max_res=max(max_res,float(np.max(abs(pred-x[1:]))))
  acc=(-20000*(x[:,0]-x[:,2])-2000*(x[:,1]-x[:,3])+u)/m
  max_acc=max(max_acc,float(np.max(abs(acc-z['y'][:,0]))));maxforce=max(maxforce,float(np.max(abs(u))))
  if 'NN' in path.name:mincontact=min(mincontact,float(core.metrics(z)['min_contact_ratio']))
  if path.name.endswith('NN_fixed.npz'):
   assert np.all(z['estimate']==300)
   with np.load(path.with_name(path.name.replace('NN_fixed','NN_oracle'))) as other:
    np.testing.assert_array_equal(road,other['road']);np.testing.assert_array_equal(z['mass'],other['mass'])
assert count==1260 and max_res<1e-10 and max_acc<1e-10 and maxforce<=500+1e-9
extra={};worst=[]
for kind,g in s.items():
 extra[kind]={}
 for delta,v in g.items():
  d=float(delta);sub=[r for r in rows if r['kind']==kind and r['delta']==d and r['method']=='NN_fixed']
  omission=[];change=[];advantages=[]
  for r in sub:
   o=lookup[kind,r['seed'],d,'NN_oracle'];p=lookup[kind,r['seed'],d,'passive'];nom=lookup[kind,r['seed'],0.,'NN_fixed'];pn=lookup[kind,r['seed'],0.,'passive']
   omission.append(r['acc_rms']-o['acc_rms']);advantages.append(100*(nom['acc_rms']/pn['acc_rms']-r['acc_rms']/p['acc_rms']))
   worst.append(dict(kind=kind,seed=r['seed'],delta=d,penalty_pct=100*(r['acc_rms']/o['acc_rms']-1),fixed_rms=r['acc_rms'],oracle_rms=o['acc_rms']))
  extra[kind][delta]=dict(omission_absolute_rms=interval(omission),passive_advantage_change_pp=interval(advantages),ratio_of_means_omission_pct=100*(v['mean_acc_rms']['NN_fixed']/v['mean_acc_rms']['NN_oracle']-1))
validation=dict(rollouts=count,max_transition_residual=max_res,max_acceleration_residual=max_acc,max_force_N=maxforce,min_NN_contact_ratio=mincontact,
 mild_worst_relative_omission=max((w for w in worst if abs(w['delta'])<=.1),key=lambda w:w['penalty_pct']),extra=extra)
(OUT/'validation.json').write_text(json.dumps(validation,indent=2))
print(json.dumps({k:v for k,v in validation.items() if k!='extra'},indent=2))
