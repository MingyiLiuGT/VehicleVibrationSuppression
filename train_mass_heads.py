"""Mass-conditioned neural policy with shared features and paired action heads.

The scalar mass input interpolates learned action heads; all weights stay fixed
online. No MPC optimization or force-correction controller runs at inference.
"""
import bootstrap
from bootstrap import ROOT
import numpy as np,json,time,argparse
import core
from mass_optimization import OUT,savecsv
from train_paired_mass import GRID,NOM
from threadpoolctl import threadpool_limits

def forward(x,w,b):
 a=[x]
 for wi,bi in zip(w[:-1],b[:-1]):a.append(np.maximum(a[-1]@wi+bi,0))
 return a[-1]@w[-1]+b[-1],a

def metrics(x,y,w,b):
 raw=forward(x,w,b)[0];p=np.clip(raw,-1,1);delta=(p-p[:,NOM,None])-(y-y[:,NOM,None])
 return dict(action_rmse_N=float(500*np.sqrt(np.mean((p-y)**2))),nominal_rmse_N=float(500*np.sqrt(np.mean((p[:,NOM]-y[:,NOM])**2))),mass_delta_rmse_N=float(500*np.sqrt(np.mean(delta[:,np.arange(len(GRID))!=NOM]**2))),loss=float(np.mean((raw-y)**2)+25*np.mean(delta**2)))

def train(source='paired_dense',name='mass_heads_dense',seed=17001,width=128):
 out=OUT/name;out.mkdir(exist_ok=True);z=np.load(OUT/source/'paired_dataset.npz');base=z['train_x'][:,0,:-1];mean=base.mean(0);scale=np.maximum(base.std(0),1e-6);x=(base-mean)/scale;val=(z['validation_x'][:,0,:-1]-mean)/scale;test=(z['test_x'][:,0,:-1]-mean)/scale;y=z['train_y']
 rng=np.random.default_rng(seed);dims=[x.shape[1],width,width,len(GRID)];w=[rng.normal(0,np.sqrt(2/a),(a,b)) for a,b in zip(dims[:-1],dims[1:])];b=[np.zeros(d) for d in dims[1:]]
 # Initialize all heads at the previous policy's nominal input, preserving its existing control map.
 old=np.load(ROOT/'results/models/policy.npz')
 w=[old['w0'][:-1].copy()*(scale/old['scale'][:-1])[:,None],old['w1'].copy(),np.repeat(old['w2'],len(GRID),axis=1)]
 b=[old['b0'].copy()+(mean-old['mean'][:-1])/old['scale'][:-1]@old['w0'][:-1]+(300-old['mean'][-1])/old['scale'][-1]*old['w0'][-1],old['b1'].copy(),np.repeat(old['b2'],len(GRID))]
 params=w+b;mom=[np.zeros_like(p) for p in params];vel=[np.zeros_like(p) for p in params];step=0;best=np.inf;wait=0;history=[];start=time.perf_counter()
 for ep in range(1,2001):
  order=rng.permutation(len(x));lr=.001*(.3 if ep>500 else 1)*(.3 if ep>1000 else 1)*(.3 if ep>1500 else 1)
  for starti in range(0,len(x),128):
   ids=order[starti:starti+128];p,a=forward(x[ids],w,b);yy=y[ids];ap=np.clip(p,-1,1);delta=(ap-ap[:,NOM,None])-(yy-yy[:,NOM,None]);gd=50*delta/p.size;gd[:,NOM]-=gd.sum(1);g=2*(p-yy)/p.size+gd*((p>-1)&(p<1));gw=[None]*3;gb=[None]*3
   for j in [2,1,0]:
    gw[j]=a[j].T@g+1e-7*w[j];gb[j]=g.sum(0)
    if j:g=(g@w[j].T)*(a[j]>0)
   step+=1
   for pp,mm,vv,gg in zip(params,mom,vel,gw+gb):
    mm*=.9;mm+=.1*gg;vv*=.999;vv+=.001*gg**2;pp-=lr*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8)
  sc=metrics(val,z['validation_y'],w,b);history.append(dict(epoch=ep,**sc))
  if sc['loss']<best-1e-10:best=sc['loss'];bestp=([p.copy() for p in w],[p.copy() for p in b]);bestep=ep;wait=0
  else:wait+=1
  if ep%100==0:print(name,ep,sc,flush=True);savecsv(out/'training_history.csv',history)
  if ep>=1100 and wait>=200:break
 w,b=bestp;np.savez(out/'policy.npz',mean=mean,scale=scale,grid=GRID,**{f'w{i}':p for i,p in enumerate(w)},**{f'b{i}':p for i,p in enumerate(b)})
 report=dict(source=source,architecture=dims,mass_input='Scalar mass gates piecewise-linear interpolation among learned heads; clipped to training endpoints. This is part of the network inference.',seed=seed,best_epoch=bestep,epochs=ep,seconds=time.perf_counter()-start,validation=metrics(val,z['validation_y'],w,b),test=metrics(test,z['test_y'],w,b))
 (out/'training.json').write_text(json.dumps(report,indent=2));savecsv(out/'training_history.csv',history);print(report,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='mass_heads_dense');ap.add_argument('--source',default='paired_dense');ap.add_argument('--seed',type=int,default=17001);a=ap.parse_args()
 with threadpool_limits(limits=1):train(a.source,a.name,a.seed)
