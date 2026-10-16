"""Constant true mass 400 kg; compare 300/400 kg prediction models at ±500 N."""
import bootstrap
from bootstrap import ROOT
import core
import numpy as np
import json, hashlib
from mass_optimization import savecsv
from threadpoolctl import threadpool_limits

OUT = ROOT / 'results/constant400_mpc'

def main():
    OUT.mkdir(exist_ok=True)
    (OUT/'traces').mkdir(exist_ok=True)
    roads = {'sine': range(16000,16012), 'multisine': range(16100,16112)}
    windows = {'whole':(0,24), 'interior':(4,20)}
    protocol = dict(true_mass_kg=400, prediction_masses_kg=[300,400], force_limit_N=500,
                    duration_s=24, windows_s=windows, roads={k:list(v) for k,v in roads.items()},
                    initial_state=[0,0,0,0], horizon=30, dt=.01,
                    description='Constant mass from t=0; no mass switch or identification. Same MPC objective, exact state and preview. Primary window 0–24 s; 4–20 s secondary.',
                    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['core.py','evaluate_constant400_mpc.py']})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]; residual=0.; maxforce=0.; maxkkt=0.
    for kind,seeds in roads.items():
        for seed in seeds:
            for model,method in [(300,'MPC_fixed'),(400,'MPC_oracle')]:
                tr=core.rollout(kind,seed,method=method,duration=24,mass=400,limit=500)
                assert np.all(tr['mass']==400) and np.all(tr['estimate']==model)
                a,b,e=core.discrete(400)
                err=np.max(abs(tr['x'][1:]-(tr['x'][:-1]@a.T+tr['u'][:-1,None]*b+tr['road'][:-1,None]*e)))
                residual=max(residual,float(err)); maxforce=max(maxforce,float(max(abs(tr['u'])))); maxkkt=max(maxkkt,float(max(tr['kkt'])))
                assert err<1e-10 and max(abs(tr['u']))<=500+1e-7
                np.savez_compressed(OUT/'traces'/f'{kind}_{seed}_model{model}.npz',**tr)
                for window,(start,end) in windows.items():
                    mask=(tr['t']>=start)&(tr['t']<end)
                    sub={k:v[mask] for k,v in tr.items()}
                    y=sub['y'];u=sub['u']
                    cost=np.mean(y[:,0]**2+.02*(y[:,1]/core.TRAVEL_SCALE)**2+.02*(y[:,2]/core.LOAD_SCALE)**2+core.EFFORT*(u/core.FORCE_SCALE)**2)
                    rows.append(dict(kind=kind,seed=seed,model_mass=model,window=window,realized_stage_cost=float(cost),**core.metrics(sub)))
    savecsv(OUT/'per_road.csv',rows)
    summary=[]
    for kind in roads:
        for window in windows:
            rr=[r for r in rows if r['kind']==kind and r['window']==window]
            means={str(m):{k:float(np.mean([r[k] for r in rr if r['model_mass']==m])) for k in ['acc_rms','travel_rms_mm','tire_rms_N','force_rms_N','saturation_pct','realized_stage_cost']} for m in [300,400]}
            by={m:{r['seed']:r['acc_rms'] for r in rr if r['model_mass']==m} for m in [300,400]}
            summary.append(dict(kind=kind,window=window,means=means,model400_acc_reduction_pct=100*(1-means['400']['acc_rms']/means['300']['acc_rms']),model400_better_roads=sum(by[400][s]<by[300][s] for s in by[300]),roads=12))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'validation.json').write_text(json.dumps(dict(trajectories=48,constant_mass_and_model_checks_pass=True,max_transition_residual=residual,max_force_N=maxforce,max_kkt_residual=maxkkt),indent=2))
    lines=['# Constant 400 kg plant: 300 vs 400 kg MPC at ±500 N','','24 roads, 48 trajectories. Primary window: complete 0–24 s. Positive reduction favors the 400 kg model. No identification or mass switching.','','| Road | Window | 300 kg model RMS | 400 kg model RMS | 400 kg reduction (%) | 400 kg better roads |','|---|---|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['kind']} | {s['window']} | {s['means']['300']['acc_rms']:.6f} | {s['means']['400']['acc_rms']:.6f} | {s['model400_acc_reduction_pct']:.3f} | {s['model400_better_roads']}/12 |")
        print(json.dumps(s),flush=True)
    lines+=['','RMS units: m/s². Interior window is 4–20 s. Additional suspension travel, tire load, saturation and realized multiobjective stage cost are in summary.json and per_road.csv. The realized cost uses actual plant outputs with the unchanged core.MPC weights; it is not a finite-horizon predicted optimal cost.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
