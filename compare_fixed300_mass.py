"""Paired plant-mass comparison with the existing policy's mass input fixed at 300 kg.

This policy was trained across 240, 300, 360, 420 kg. It is not a newly trained
300-kg-only network. The 200-kg plant is outside that training mass range.
"""
import bootstrap
from bootstrap import ROOT
from core import Policy,rollout,metrics
import numpy as np
import csv,json,hashlib

out=ROOT/'results'/'fixed300_mass_comparison'
out.mkdir(exist_ok=True)
policy=Policy(ROOT/'results/models/policy.npz')
rows=[];summary={}
for kind,base in [('sine',4000),('multisine',4200),('bump',4300)]:
    pairs=[]
    for seed in range(base,base+8):
        pair=[];roads=[]
        for mass in [300.,200.]:
            tr=rollout(kind,seed,'NN_fixed',mass=mass,duration=6.,policy=policy)
            assert np.all(tr['estimate']==300.)
            roads.append(tr['road']);v=metrics(tr);pair.append(v)
            rows.append(dict(road_kind=kind,seed=seed,plant_mass_kg=mass,policy_mass_input_kg=300.,**v))
            np.savez_compressed(out/f'{kind}_{seed}_{int(mass)}kg.npz',**tr)
        assert np.array_equal(*roads)
        pairs.append(pair)
    data={}
    for key in ['acc_rms','acc_peak','saturation_pct']:
        v=np.array([[p[0][key],p[1][key]] for p in pairs])
        relative=np.divide(v[:,1]-v[:,0],v[:,0],out=np.full(len(v),np.nan),where=v[:,0]!=0)*100
        valid=relative[np.isfinite(relative)]
        data[key]={'mean_300kg':float(v[:,0].mean()),'mean_200kg':float(v[:,1].mean()),
                   'ratio_of_means_change_pct':float(100*(v[:,1].mean()/v[:,0].mean()-1)),
                   'paired_change_min_pct':float(valid.min()) if len(valid) else None,
                   'paired_change_max_pct':float(valid.max()) if len(valid) else None}
    summary[kind]=data
with (out/'per_road.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
report={'protocol':{'policy':'existing mass-conditioned policy; mass input fixed at 300 kg','trained_mass_grid_kg':[240,300,360,420],
    'plant_mass_kg':[300,200],'road_seeds':{'sine':[4000,4007],'multisine':[4200,4207],'bump':[4300,4307]},
    'duration_s':6,'dt_s':.01,'force_limit_N':500,'preview_s':.3,'initial_state':'zero','preview':'exact',
    'model_sha256':hashlib.sha256((ROOT/'results/models/policy.npz').read_bytes()).hexdigest()},'summary':summary}
(out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(summary,indent=2))
