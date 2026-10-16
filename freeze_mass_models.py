"""Freeze the declared mass-head model and two ablations without overwriting data.

Run after paired-dense training and development-only estimator selection.
Existing files are checked, never silently replaced. This is packaging, not a
search over held-out control results.
"""
import bootstrap
from bootstrap import ROOT
from mass_optimization import OUT
import numpy as np,json,shutil,hashlib

def main():
 training=json.loads((OUT/'mass_heads_dense/training.json').read_text())
 estimator=json.loads((OUT/'estimator_selection.json').read_text())['selected']
 refined=OUT/'mass_heads_dense/policy.npz'
 spec={
  'mass_heads_final':(refined,dict(training,estimator=estimator,selection='Mass-head dense network selected for lower validation action and mass-difference errors; estimator selected solely for mean development identification error. Original MPC objective and ideal actuator retained. Final road tests not inspected before freeze.')),
  'mass_heads_old_estimator':(refined,dict(training,selection='Ablation only: same frozen final network with original RLS forgetting/excitation threshold and expanded bounds.')),
  'original_reference':(ROOT/'results/models/policy.npz',{'source':'original policy, unchanged','note':'Same extended projection for a paired comparison; 200 kg is extrapolation for this original network.'}),
 }
 for name,(source,metadata) in spec.items():
  dest=OUT/name;dest.mkdir(exist_ok=True);path=dest/'policy.npz'
  if path.exists():
   a=np.load(source);b=np.load(path)
   assert set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),f'{name}: different model; use a new result directory'
  else:shutil.copy2(source,path)
  if name!='original_reference':metadata['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  meta=dest/'training.json'
  if meta.exists():
   old=json.loads(meta.read_text())
   for key in ['architecture','best_epoch','estimator']:
    assert old.get(key)==metadata.get(key),(name,key)
  else:meta.write_text(json.dumps(metadata,indent=2))
  print(name,hashlib.sha256(path.read_bytes()).hexdigest())
if __name__=='__main__':main()
