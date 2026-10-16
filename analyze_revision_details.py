"""Small reviewer clarifications from archived outputs; no model or data changes."""
import bootstrap
from bootstrap import ROOT
import csv
import json
import numpy as np
from core import Policy
from mass_controller import MassConditionedPolicy

OUT = ROOT / 'results'

def exceedance(raw):
    excess = np.maximum(np.abs(np.asarray(raw).ravel()) - 500., 0.)
    positive = excess[excess > 0]
    return dict(samples=int(excess.size), exceeding_samples=int(positive.size),
                raw_over_pct=float(100 * np.mean(excess > 0)),
                conditional_mean_excess_N=float(positive.mean()) if positive.size else 0.,
                maximum_excess_N=float(excess.max()),
                maximum_raw_force_N=float(np.max(np.abs(raw))))

def main():
    raw = {}
    data = np.load(OUT / 'dataset.npz')
    raw['heldout_seed_33'] = exceedance(Policy(OUT / 'models/policy.npz').predict(data['test_x']))
    for group in ['in_domain','unseen_sine','unseen_mass','bump','preview_noise','preview_delay']:
        paths = sorted((OUT / 'traces').glob(f'{group}_*_NN.npz'))
        assert len(paths) == 8
        raw[group] = exceedance(np.concatenate([np.load(p)['raw'] for p in paths]))
    for old in json.loads((OUT / 'action_errors.json').read_text()):
        if old['group'] in raw:
            assert abs(raw[old['group']]['raw_over_pct'] - old['raw_violation_pct']) < 1e-10
    data = np.load(OUT / 'mass_optimization/paired_dense/paired_dataset.npz')
    raw['refined_paired'] = exceedance(MassConditionedPolicy().predict(data['test_x'].reshape(-1,35)))
    tracking = []
    for kind in ['sine','multisine']:
        for mass in [200,240,360,420]:
            rows = []
            paths=sorted((OUT / 'mass_optimization/mass_heads_final/heldout_traces').glob(f'{kind}_*_{mass}_NN_adaptive.npz'))
            assert len(paths)==12
            for p in paths:
                tr = np.load(p);t = tr['t'];err = 100 * abs(tr['estimate']-tr['mass'])/tr['mass']
                rows.append({name:float(err[(t>=a)&(t<b)].mean()) for name,a,b in
                             [('early_error_pct',4,8),('terminal_error_pct',18,20),('return_terminal_error_pct',22,24)]})
            tracking.append(dict(kind=kind,mass_kg=mass,roads=12,
                                 **{key:float(np.mean([r[key] for r in rows])) for key in rows[0]}))
    tr=np.load(OUT / 'mass_optimization/mass_heads_final/heldout_traces/sine_16001_200_NN_adaptive.npz')
    t=tr['t'];err=100*abs(tr['estimate']-tr['mass'])/tr['mass']
    representative={name:float(err[(t>=a)&(t<b)].mean()) for name,a,b in
                    [('early_error_pct',4,8),('terminal_error_pct',18,20)]}
    rows=list(csv.DictReader((OUT / 'closed_loop.csv').open()))
    bump={}
    for method in ['FFPD','MPC_no_preview','MPC','NN']:
        rr=[r for r in rows if r['group']=='bump' and r['method']==method]
        bump[method]={k:float(np.mean([float(r[k]) for r in rr])) for k in
                      ['acc_rms','acc_peak','travel_rms_mm','tire_rms_N','force_rms_N','saturation_pct']}
    methods=json.loads((OUT / 'mass_optimization/timing/summary.json').read_text())['methods']
    timing={mode+'_mpc_to_nn_mean_ratio':methods['MPC_'+mode]['total_ms']['mean']/methods['NN_'+mode]['total_ms']['mean']
            for mode in ['fixed','adaptive']}
    report=dict(raw_exceedance_definition='max(abs(raw force)-500 N,0); conditional mean uses exceeding samples only',
                raw_exceedance=raw,tracking_definition='Mean absolute relative mass error per road, then arithmetic mean across roads; early 4–8 s, terminal 18–20 s, return terminal 22–24 s. No convergence-time claim.',
                mass_tracking=tracking,representative_mass_tracking=representative,bump=bump,timing=timing,
                measured_road=dict(required_file='code/quarter_car/w_all_reversed_1.npy',
                                   available=(ROOT.parent/'code/quarter_car/w_all_reversed_1.npy').exists(),
                                   status='Original local array remains absent. A new public P79 excerpt from the same source is revalidated with frozen quarter-car policies and a four-DOF half-car MPC; see road_halfcar/source/provenance.json.',
                                   public_excerpt_revalidated=(OUT/'road_halfcar/validation.json').exists()))
    (OUT / 'revision_details.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
