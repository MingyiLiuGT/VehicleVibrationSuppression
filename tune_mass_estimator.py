"""Choose RLS excitation gate/forgetting using development identification error only."""
import bootstrap
import numpy as np,json
import core
from mass_optimization import OUT,savecsv
class TunedRLS(core.MassRLS):
 forget_value=.995;gate=.01
 def __init__(self):super().__init__(self.forget_value)
 def update(self,x,u,measured_acc):
  phi=(-core.KS*(x[0]-x[2])-core.CS*(x[1]-x[3])+u)/300
  if abs(phi)<self.gate:return self.mass
  gain=self.p*phi/(self.forget+phi*phi*self.p)
  self.theta=float(np.clip(self.theta+gain*(measured_acc-phi*self.theta),300/480,300/180));self.p=float(np.clip((self.p-gain*phi*self.p)/self.forget,1e-8,1e4));self.updates+=1
  return self.mass

def tune():
 rows=[];files=sorted((OUT/'mass_heads_dense/development_traces').glob('*NN_fixed.npz'));assert len(files)==32
 protocol={'forgetting':[.97,.99,.995,.999],'gates':[.005,.02,.05],'selection':'Mean absolute relative mass error over all 4–24 s development fixed-input trajectories. Closed-loop gain is not a selection criterion. No held-out trajectories used.'};(OUT/'estimator_tuning_protocol.json').write_text(json.dumps(protocol,indent=2))
 for forget in protocol['forgetting']:
  for gate in protocol['gates']:
   TunedRLS.forget_value=forget;TunedRLS.gate=gate
   for path in files:
    tr=dict(np.load(path));seed=int(path.stem.split('_')[1]);rng=np.random.default_rng(seed+80000);rng.normal(0,0,len(tr['t'])+101);est=TunedRLS();errors=[]
    for k in range(len(tr['t'])):
     errors.append(abs(est.mass-tr['mass'][k])/tr['mass'][k]*100);est.update(tr['x'][k],tr['u'][k],tr['y'][k,0]+rng.normal(0,.05))
    err=np.array(errors);rows.append(dict(forget=forget,gate=gate,trajectory=path.stem,error_pct=float(err[tr['t']>=4].mean()),loaded_error_pct=float(err[(tr['t']>=4)&(tr['t']<20)].mean()),recovery_error_pct=float(err[tr['t']>=20].mean())))
 candidates=[]
 for forget in protocol['forgetting']:
  for gate in protocol['gates']:
   rr=[r for r in rows if r['forget']==forget and r['gate']==gate];candidates.append(dict(forget=forget,gate=gate,error_pct=float(np.mean([r['error_pct'] for r in rr])),loaded_error_pct=float(np.mean([r['loaded_error_pct'] for r in rr])),recovery_error_pct=float(np.mean([r['recovery_error_pct'] for r in rr]))))
 best=min(candidates,key=lambda r:r['error_pct']);savecsv(OUT/'estimator_tuning.csv',rows);(OUT/'estimator_selection.json').write_text(json.dumps(dict(selected=best,candidates=candidates),indent=2));print(best,flush=True)
if __name__=='__main__':tune()
