"""Retain all 48 original synthetic cases when checking the refined road policy."""
import bootstrap
from bootstrap import ROOT
import json
import numpy as np
from threadpoolctl import threadpool_limits
import core
from mass_controller import MassConditionedPolicy
from mass_optimization import savecsv
from refine_road_coverage import INITIAL_POLICY

def main():
 out=ROOT/'results/road_coverage_centered';rows=[]
 cases=[('in_domain','sine',300.,0.,0),('unseen_sine','unseen_sine',300.,0.,0),('unseen_mass','multisine',330.,0.,0),('bump','bump',300.,0.,0),('preview_noise','multisine',300.,.002,0),('preview_delay','multisine',300.,0.,-2)]
 for ci,(group,kind,mass,noise,shift) in enumerate(cases):
  for rep in range(8):
   seed=4000+ci*100+rep
   extra=dict(frequency=3.4+.1*rep,amplitude=.018) if kind=='unseen_sine' else {}
   for name,path in [('prior_refined',INITIAL_POLICY),('refined',out/'policy.npz')]:
    tr=core.rollout(kind,seed,'NN',mass=mass,policy=MassConditionedPolicy(path),preview_noise=noise,preview_shift=shift,**extra)
    met=core.metrics(tr);assert np.max(abs(tr['u']))<=500 and met['min_contact_ratio']>0
    rows.append(dict(group=group,seed=seed,method=name,**met))
  print(group,flush=True)
 savecsv(out/'synthetic_regression.csv',rows)
 summary={g:{n:float(np.mean([r['acc_rms'] for r in rows if r['group']==g and r['method']==n])) for n in ['prior_refined','refined']} for g,*_ in cases}
 (out/'synthetic_regression.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
