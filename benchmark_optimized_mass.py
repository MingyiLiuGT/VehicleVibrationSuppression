"""Chronological replay of four full software paths on one host."""
import bootstrap
from bootstrap import ROOT
import core,numpy as np,json,time,platform,csv,gc,argparse
from pathlib import Path
from mass_controller import MassConditionedPolicy,MassEstimator
from mass_optimization import OUT,savecsv
from threadpoolctl import threadpool_limits

def main(output_dir=None):
 out=Path(output_dir) if output_dir else OUT/'timing';out.mkdir(parents=True,exist_ok=True);p=MassConditionedPolicy();methods=['NN_fixed','NN_adaptive','MPC_fixed','MPC_adaptive'];records=[];errors=[];rebuilds=0
 # Initialization sampled separately.
 init=[]
 for _ in range(20):
  t=time.perf_counter_ns();core.MPC(300);init.append((time.perf_counter_ns()-t)/1e6)
 warm=core.MPC(300)
 for _ in range(100):warm.solve(np.zeros(4),np.zeros(30));p.action(np.zeros(4),np.zeros(30),300)
 count=0
 for kind,seed,mass in [('sine',16001,200),('multisine',16100,420),('sine',16000,200)]:
  tr=dict(np.load(OUT/f'mass_heads_final/heldout_traces/{kind}_{seed}_{mass}_NN_adaptive.npz'));_,road=core.road_profile(kind,seed,24);rng=np.random.default_rng(seed+80000);rng.normal(0,0,len(tr['t'])+101);noise=rng.normal(0,.05,len(tr['t']));estimators={m:MassEstimator() for m in methods};qps={'MPC_fixed':core.MPC(300),'MPC_adaptive':core.MPC(300)};last_mass=300.
  for k in range(len(tr['t'])):
   for method in methods[count%4:]+methods[:count%4]:
    cpu0=time.thread_time_ns()
    t0=time.perf_counter_ns();preview=road[k+np.arange(30)].copy();t1=time.perf_counter_ns();est=estimators[method]
    if 'adaptive' in method and k:est.update(tr['x'][k-1],tr['u'][k-1],tr['y'][k-1,0]+noise[k-1])
    m=est.mass if 'adaptive' in method else 300.;t2=time.perf_counter_ns()
    if method=='MPC_adaptive':
     clipped=float(np.clip(m,200,450))
     if clipped!=last_mass:qps[method]=core.MPC(clipped);last_mass=clipped;rebuilds+=1
    t3=time.perf_counter_ns()
    if method.startswith('NN'):u,_=p.action(tr['x'][k],preview,m)
    else:u,_=qps[method].solve(tr['x'][k],preview)
    t4=time.perf_counter_ns();cpu1=time.thread_time_ns();records.append(dict(method=method,index=count,preview_ms=(t1-t0)/1e6,id_ms=(t2-t1)/1e6,model_ms=(t3-t2)/1e6,control_ms=(t4-t3)/1e6,total_ms=(t4-t0)/1e6,thread_cpu_ms=(cpu1-cpu0)/1e6,wall_minus_cpu_ms=((t4-t0)-(cpu1-cpu0))/1e6))
    if method=='NN_adaptive':errors.append(abs(u-tr['u'][k]))
   count+=1
   if count>=5000:break
  if count>=5000:break
 savecsv(out/'samples.csv',records)
 summary={method:{key:{'mean':float(np.mean(v:=[r[key] for r in records if r['method']==method])),'p95':float(np.percentile(v,95)),'max':float(np.max(v))} for key in ['preview_ms','id_ms','model_ms','control_ms','total_ms']} for method in methods}
 report=dict(platform=platform.platform(),python=platform.python_version(),calls_per_path=count,warmup=100,methods=summary,adaptive_mpc_rebuilds=rebuilds,initialization_mean_ms=float(np.mean(init)),max_nn_replay_error_N=float(max(errors)),garbage_collection_enabled=gc.isenabled(),timer='perf_counter_ns wall time; thread_time_ns brackets each call outside the wall-time interval for diagnostics only',note='Timing includes preview assembly, causal previous-sample RLS, mass selection, network input normalization, learned head interpolation, force projection; adaptive MPC includes every changed clipped-mass rebuild. Sensing, state/road reconstruction and physical actuation excluded.')
 assert max(errors)<1e-9
 (out/'summary.json').write_text(json.dumps(report,indent=2));print({m:v['total_ms'] for m,v in summary.items()},flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output-dir');ap.add_argument('--disable-gc',action='store_true');args=ap.parse_args()
 was_enabled=gc.isenabled()
 if args.disable_gc:gc.collect();gc.disable()
 try:
  with threadpool_limits(limits=1):main(args.output_dir)
 finally:
  if was_enabled:gc.enable()
