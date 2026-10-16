"""Numerical validation of generated results and data provenance."""
import bootstrap
from bootstrap import ROOT
import csv,json,hashlib
import numpy as np
from core import *

out=ROOT/'results'; d=np.load(out/'dataset.npz')
sets=[set(d[s+'_scenario']) for s in ['train','validation','test']]
assert not any(sets[i]&sets[j] for i,j in [(0,1),(0,2),(1,2)])
checked=0;largest=0;maxdyn=0
for p in (out/'traces').glob('*.npz'):
    with np.load(p) as data:
        tr={key:data[key] for key in data.files}
    if 'u' not in tr:continue
    for key in ['x','u','y','mass']:assert np.isfinite(tr[key]).all(),(p,key)
    # Sweeps are saved in tables; all archived standard rollouts use 500 N.
    assert max(abs(tr['u']))<=500+1e-6,p
    if tr['kkt'].size:largest=max(largest,float(max(tr['kkt'])))
    plants={float(m):discrete(float(m)) for m in set(tr['mass'])}
    for k in range(len(tr['t'])-1):
        ad,bd,ed=plants[tr['mass'][k]]
        nxt=ad@tr['x'][k]+bd*tr['u'][k]+ed*tr['road'][k]
        maxdyn=max(maxdyn,float(np.max(abs(nxt-tr['x'][k+1]))))
    checked+=1
assert largest<1e-4,largest
assert maxdyn<1e-10,maxdyn
report=dict(scenario_disjoint=True,checked_rollouts=checked,max_projected_kkt=largest,max_transition_residual=maxdyn,
            force_bound_N=500,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.glob('*.json')) if p.name!='validation.json'})
(out/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))

# Verify the complete-path timing aggregates without remeasuring this host.
rows=list(csv.DictReader((out/'mass_optimization/timing/samples.csv').open()))
stats=json.loads((out/'mass_optimization/timing/summary.json').read_text())['methods']
for method,summary in stats.items():
    part=[r for r in rows if r['method']==method]
    assert len(part)==5000
    times=np.array([float(r['total_ms']) for r in part])
    for key,value in [('mean',times.mean()),('p95',np.percentile(times,95)),('max',times.max())]:
        assert abs(value-summary['total_ms'][key])<1e-12
    for r in part:
        assert abs(float(r['total_ms'])-sum(float(r[k]) for k in ['preview_ms','id_ms','model_ms','control_ms']))<1e-12
print('All 20,000 archived complete-path timing samples and aggregates verified')
# Independently recompute the supplementary fixed/true-mass comparisons.
sensitivity=json.loads((out/'mass_mpc_comparison/summary.json').read_text())
rows=list(csv.DictReader((out/'mass_mpc_comparison/metrics.csv').open()))
for kind,groups in sensitivity['groups'].items():
    for delta,models in groups.items():
        for model,values in models.items():
            fixed={int(r['seed']):float(r['acc_rms']) for r in rows if r['study']=='constant' and r['kind']==kind and float(r['delta'])==float(delta) and r['method']==model+'_fixed'}
            oracle={int(r['seed']):float(r['acc_rms']) for r in rows if r['study']=='constant' and r['kind']==kind and float(r['delta'])==float(delta) and r['method']==model+'_oracle'}
            assert len(fixed)==len(oracle)==20
            assert abs(np.mean(list(fixed.values()))-values['fixed_mean_rms'])<1e-12
            assert abs(np.mean([100*(1-oracle[k]/fixed[k]) for k in fixed])-values['gain_pct']['mean'])<1e-12
print('42 fixed/true-mass statistical groups verified')
