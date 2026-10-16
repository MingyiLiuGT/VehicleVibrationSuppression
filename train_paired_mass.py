"""Paired mass-conditioned NN distillation; same state/preview across mass labels.

Uses scenario-disjoint original train/validation/action-test sources. Optimizer
selects checkpoints on action and paired mass-difference error, not closed-loop
adaptation gain. An explicit 300-kg input ablation uses the SAME trained network.
"""
import bootstrap
from bootstrap import ROOT
import core,numpy as np,json,csv,argparse,time
from threadpoolctl import threadpool_limits
from mass_optimization import OUT,savecsv
GRID=np.array([200.,240.,270.,300.,330.,360.,420.,450.]);NOM=3

def dataset(name,weights,effort,stride=2):
 out=OUT/name;out.mkdir(exist_ok=True)
 core.WEIGHTS=np.array(weights);core.EFFORT=effort
 teachers=[core.MPC(m) for m in GRID];old=np.load(ROOT/'results/dataset.npz');data={}
 for split in ['train','validation','test']:
  # Striding within each road retains all scenario IDs; group split remains road-disjoint.
  source=old[split+'_x'][::stride];x=np.repeat(source[:,None,:],len(GRID),axis=1);x[:,:,-1]=GRID
  y=np.empty(x.shape[:2])
  for i,xx in enumerate(x):
   for j,(features,qp) in enumerate(zip(xx,teachers)):y[i,j]=qp.solve(features[:4],features[4:-1])[0]/500
  data[split+'_x']=x;data[split+'_y']=y;data[split+'_scenario']=old[split+'_scenario'][::stride]
  print(name,'labels',split,x.shape,flush=True)
 np.savez_compressed(out/'paired_dataset.npz',**data)
 metadata=dict(masses=GRID.tolist(),weights=weights,effort=effort,stride=stride,
  paired_source=f'Every {stride}th source sample, with original scenario-disjoint splits; identical state and preview labeled at all eight masses',
  training='Network training is defined separately in train_mass_heads.py and mass_heads_dense/training.json')
 (out/'protocol.json').write_text(json.dumps(metadata,indent=2))
 return out

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--name',default='paired_dense');a=ap.parse_args()
 with threadpool_limits(limits=1):dataset(a.name,[1.,.02,.02],.005,stride=2)
