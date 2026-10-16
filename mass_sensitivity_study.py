"""Separate physical mass sensitivity from the value of providing mass to the NN.

Frozen policy, paired new roads, no retraining or parameter selection. Percent
changes are descriptive; this is not a universal stability/robustness proof.
"""
import bootstrap
from bootstrap import ROOT
import core
import numpy as np
import json,csv,hashlib
from pathlib import Path
OUT=ROOT/'results/mass_sensitivity';OUT.mkdir(exist_ok=True)
DELTAS=[-.20,-.10,-.05,0.,.05,.10,.20]
def interval(values):
    x=np.asarray(values);boot=np.random.default_rng(98765).choice(x,(10000,len(x)),replace=True).mean(axis=1)
    return dict(mean=float(x.mean()),ci95=list(np.percentile(boot,[2.5,97.5])),minimum=float(x.min()),maximum=float(x.max()),n=len(x))
def main():
    model=ROOT/'results/models/policy.npz'
    protocol=dict(policy_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      status='written before evaluation',mass_deltas=DELTAS,nominal_mass_kg=300,
      seeds={k:list(range(b,b+20)) for k,b in [('multisine',11000),('sine',11100),('bump',11200)]},duration_s=6,
      mild_range='±5% and ±10%; ±20% reported as stress cases',
      comparisons=['actual mass changes with NN mass input frozen at 300 kg','fixed versus true-mass input on identical actual plant','same-state action response to mass input','NN versus passive at the same actual mass'],
      assumptions='Known state; exact preview; unchanged stiffness/damping; constant actual mass for each run; no sensor noise in NN inputs.',
      statistics='Road-level paired changes; 10000 bootstrap resamples per group; no pooling across masses; no formal engineering equivalence tolerance or safety guarantee.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    policy=core.Policy(model);rows=[];action_rows=[];summary={};td=OUT/'traces';td.mkdir(exist_ok=True)
    for kind,seeds in protocol['seeds'].items():
      road_data={}
      for seed in seeds:
        results={}
        for delta in DELTAS:
          mass=300*(1+delta)
          for method in ['NN_fixed','NN_oracle','passive']:
            tr=core.rollout(kind,seed,method,mass=mass,policy=policy,duration=6)
            met=core.metrics(tr);results[(delta,method)]=met
            rows.append(dict(kind=kind,seed=seed,delta=delta,mass=mass,method=method,**met))
            np.savez_compressed(td/f'{kind}_{seed}_m{mass:.0f}_{method}.npz',**tr)
            if delta==0 and method=='NN_fixed':nominal=tr
        # Hold physical state and preview fixed; only change the mass input.
        _,road=core.road_profile(kind,seed,duration=6)
        features=np.array([core.features(x,road[i:i+30],300.) for i,x in enumerate(nominal['x'])])
        raw0=policy.predict(features);u0=np.clip(raw0,-500,500)
        for delta in DELTAS:
          xx=features.copy();xx[:,-1]=300*(1+delta);raw=policy.predict(xx);u=np.clip(raw,-500,500)
          action_rows.append(dict(kind=kind,seed=seed,delta=delta,raw_difference_rms_N=float(np.sqrt(np.mean((raw-raw0)**2))),
            applied_difference_rms_N=float(np.sqrt(np.mean((u-u0)**2))),applied_max_difference_N=float(np.max(abs(u-u0)))))
        road_data[seed]=results
      groups={}
      for delta in DELTAS:
        physical=[];omission=[];passive_benefit=[];absolute=[]
        for seed in seeds:
          r=road_data[seed];fix=r[(delta,'NN_fixed')]['acc_rms'];nom=r[(0.,'NN_fixed')]['acc_rms'];oracle=r[(delta,'NN_oracle')]['acc_rms'];pas=r[(delta,'passive')]['acc_rms']
          physical.append(100*(fix/nom-1));omission.append(100*(fix/oracle-1));passive_benefit.append(100*(1-fix/pas));absolute.append(fix-nom)
        sub=[r for r in rows if r['kind']==kind and r['delta']==delta];actions=[r for r in action_rows if r['kind']==kind and r['delta']==delta]
        groups[str(delta)]=dict(physical_mass_change_pct=interval(physical),mass_input_omission_penalty_pct=interval(omission),
          reduction_versus_passive_pct=interval(passive_benefit),absolute_acc_rms_change=interval(absolute),
          mean_acc_rms={m:float(np.mean([r['acc_rms'] for r in sub if r['method']==m])) for m in ['NN_fixed','NN_oracle','passive']},
          action_input_sensitivity=dict(mean_rms_N=float(np.mean([a['applied_difference_rms_N'] for a in actions])),max_N=max(a['applied_max_difference_N'] for a in actions)),
          min_contact_ratio=min(r['min_contact_ratio'] for r in sub if r['method']!='passive'),
          max_travel_peak_mm=max(r['travel_peak_mm'] for r in sub if r['method']!='passive'))
      summary[kind]=groups
      print(kind,json.dumps({d:{k:g[k] for k in ['physical_mass_change_pct','mass_input_omission_penalty_pct','action_input_sensitivity']} for d,g in groups.items() if d in ['-0.1','0.1']}),flush=True)
    for name,data in [('metrics.csv',rows),('action_sensitivity.csv',action_rows)]:
      with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    assert len(rows)==1260
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('Finished 1260 paired runs on 60 independent roads.',flush=True)
if __name__=='__main__':main()
