"""Revalidate a public measured profile with frozen quarter-car policies and half-car MPC.

Run normally to recompute; --check audits saved traces; --figures redraws only.
Eight predefined consecutive road windows, no retraining or test-driven tuning.
"""
import bootstrap
from bootstrap import ROOT
import argparse, csv, gzip, hashlib, json
import numpy as np
import core
import half_car as hc
from mass_controller import MassConditionedPolicy, DEFAULT_POLICY

OUT = ROOT/'results/road_halfcar'
SOURCE = OUT/'source'
METHODS = ['passive', 'FFPD', 'MPC_no_preview', 'MPC', 'NN', 'NN_refined']
START, COUNT, N = 2., 800, core.HORIZON

def roads():
    raw = gzip.decompress((SOURCE/'m13_zp_vh_first6000.csv.gz').read_bytes())
    provenance = json.loads((SOURCE/'provenance.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == provenance['excerpt_sha256']
    rows = list(csv.reader(raw.decode('cp1252').splitlines(), delimiter=';'))
    assert rows[0][21].strip() == 'Laser 21 [mm]'
    data = np.array([[float(r[0]), float(r[21])/1000] for r in rows[1:]])
    distance = data[:, 0]-data[0, 0]
    assert np.all(np.diff(distance)>0) and distance[-1]>510
    use = distance<=510
    coefficients = np.polyfit(distance[use], data[use, 1], 1)
    elevation = data[:, 1]-np.polyval(coefficients, distance)
    result = []
    t = np.arange(COUNT+N)*core.DT
    for i in range(8):
        front_s = i*60+10*t
        front = np.interp(front_s, distance, elevation)
        rear = np.interp(front_s-hc.LENGTH, distance, elevation)
        result.append((i, np.c_[front, rear]))
    np.savez_compressed(OUT/'road_inputs.npz', absolute_distance_m=data[:, 0],
        raw_right_wheelpath_m=data[:, 1], detrend_coefficients=coefficients,
        t=t, profiles=np.array([r for _, r in result]))
    return result

def rollout(profile, method, half=False):
    ad, bd, ed = hc.discrete() if half else core.discrete()
    road = profile if half else profile[:, 0]
    # Equilibrium at the initial road height avoids a fictitious entry step.
    x = np.r_[road[0].mean(), (road[0,0]-road[0,1])/hc.LENGTH,
              road[0], np.zeros(4)] if half else np.array([road[0], 0., road[0], 0.])
    qp = (hc.HalfMPC() if half else core.MPC()) if 'MPC' in method else None
    policy = (MassConditionedPolicy() if method=='NN_refined' else core.Policy(ROOT/'results/models/policy.npz')) if 'NN' in method else None
    xs, us, raws, yy, kkts = [], [], [], [], []
    for k in range(COUNT):
        preview = road[k:k+N].copy()
        if method=='MPC_no_preview': preview[:] = road[k]
        if qp:
            u, _ = qp.solve(x, preview); raw = np.array(u).copy()
            kkts.append(qp.kkt if half else qp.last['kkt'])
        elif policy: u, raw = policy.action(x, preview, 300.)
        elif method=='FFPD':
            raw = -core.KS*x[2:4]-core.CS*x[6:8] if half else -core.KS*x[2]-core.CS*x[3]
            u = np.clip(raw, -500, 500)
        else: u = raw = np.zeros(2) if half else 0.
        y = hc.outputs(x, u, road[k]) if half else core.outputs(x, u, road[k])
        xs.append(x.copy()); us.append(np.array(u).copy()); raws.append(np.array(raw).copy()); yy.append(y)
        x = ad@x+bd@u+ed@road[k] if half else ad@x+bd*u+ed*road[k]
    return dict(t=np.arange(COUNT)*core.DT, road=road[:COUNT], x=np.array(xs),
                u=np.array(us), raw=np.array(raws), y=np.array(yy), kkt=np.array(kkts),
                mass=np.full(COUNT, hc.MS if half else 300.))

def metrics(tr, half):
    if not half: return core.metrics(tr, start=START)
    sl=tr['t']>=START; y=tr['y'][sl]; u=tr['u'][sl]
    return dict(acc_rms=float(np.sqrt(np.mean(y[:, 0]**2))),
        pitch_rms=float(np.sqrt(np.mean(y[:, 1]**2))),
        travel_rms_mm=float(np.sqrt(np.mean(y[:, 2:4]**2))*1000),
        travel_peak_mm=float(abs(y[:, 2:4]).max()*1000),
        tire_rms_N=float(np.sqrt(np.mean(y[:, 4:]**2))),
        min_contact_ratio=float(np.min(1-y[:, 4:]/hc.STATIC_LOAD)),
        force_rms_N=float(np.sqrt(np.mean(u**2))),
        saturation_pct=float(np.mean(abs(u)>=495)*100),
        violation_pct=float(np.mean(abs(u)>500+1e-6)*100))

def summarize(rows):
    summary = {}
    for model in ['quarter', 'half']:
        ss = summary[model] = {}
        for method in METHODS if model=='quarter' else METHODS[:4]:
            rr=[r for r in rows if r['model']==model and r['method']==method]
            ss[method]={k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k not in ['model','segment','method']}
        ss['reductions_pct'] = {}
        for base in ['FFPD', 'MPC_no_preview', 'passive']:
            for method in ['MPC','NN','NN_refined'] if model=='quarter' else ['MPC']:
                for metric in ['acc_rms','pitch_rms'] if model=='half' else ['acc_rms']:
                    # Percentage from group means, explicitly distinct from mean per-road percentages.
                    ss['reductions_pct'][f'{method}_vs_{base}_{metric}']=100*(1-ss[method][metric]/ss[base][metric])
        fixed=[r for r in rows if r['model']==model and r['method']=='FFPD']
        ss['representative_segment']=sorted(fixed, key=lambda r:(r['acc_rms'],r['segment']))[len(fixed)//2]['segment']
    return summary

def audit():
    check=hc.validate(); errors=[]; minimum=1.; maxkkt=0.; rows=[]
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['policy_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for p in sorted((OUT/'traces').glob('*.npz')):
        model, segment, method=p.stem.split('__'); half=model=='half'
        tr=np.load(p); ad,bd,ed=hc.discrete() if half else core.discrete()
        if half: xp=tr['x'][:-1]@ad.T+tr['u'][:-1]@bd.T+tr['road'][:-1]@ed.T
        else: xp=tr['x'][:-1]@ad.T+tr['u'][:-1,None]*bd+tr['road'][:-1,None]*ed
        errors.append(float(np.max(abs(xp-tr['x'][1:]))))
        expected=np.array([(hc.outputs(x,u,r) if half else core.outputs(x,u,r)) for x,u,r in zip(tr['x'],tr['u'],tr['road'])])
        np.testing.assert_allclose(expected,tr['y'],atol=1e-12)
        assert np.max(abs(tr['u']))<=500+1e-9
        if 'NN' in method:
            policy=MassConditionedPolicy() if method=='NN_refined' else core.Policy(ROOT/'results/models/policy.npz')
            profile=np.load(OUT/'road_inputs.npz')['profiles'][int(segment),:,0]
            for k in range(COUNT):
                u,raw=policy.action(tr['x'][k],profile[k:k+N],300.)
                assert abs(u-tr['u'][k])<1e-9 and abs(raw-tr['raw'][k])<1e-9
        if tr['kkt'].size: maxkkt=max(maxkkt,float(tr['kkt'].max()))
        m=metrics(tr,half)
        contact=1-tr['y'][:,4:]/hc.STATIC_LOAD if half else 1-tr['y'][:,2]/core.LOAD_SCALE
        minimum=min(minimum,float(contact.min()))
        rows.append(dict(model=model,segment=int(segment),method=method,**m))
    assert len(rows)==80 and max(errors)<1e-10 and maxkkt<2e-5
    assert minimum>0
    stored=json.loads((OUT/'summary.json').read_text())
    fresh=summarize(rows)
    assert stored==fresh
    csv_rows=list(csv.DictReader((OUT/'per_segment.csv').open()))
    assert len(csv_rows)==len(rows)
    for saved,computed in zip(csv_rows,rows):
        for key,value in computed.items():
            if isinstance(value,str):assert saved[key]==value
            else:assert abs(float(saved[key])-value)<1e-12
    check.update(trajectories=len(rows),maximum_state_residual=max(errors),
        maximum_projected_kkt_residual=maxkkt,minimum_contact_ratio=minimum,
        applied_force_bound_N=500,quarter_neural_replay=True)
    (OUT/'validation.json').write_text(json.dumps(check,indent=2)+'\n')
    return check

def figures():
    from analyze_results import plt, finish, panel, style
    summary=json.loads((OUT/'summary.json').read_text())
    for model in ['quarter','half']:
        half=model=='half';segment=summary[model]['representative_segment']
        fig,axs=plt.subplots(2,1,figsize=(3.7,4.9),sharex=True)
        # Retain the same representative road and show the refined quarter-car NN.
        methods=['FFPD','MPC_no_preview','MPC']+([] if half else ['NN_refined'])
        for method in methods:
            tr=np.load(OUT/'traces'/f'{model}__{segment}__{method}.npz');sl=tr['t']>=START
            t=tr['t'][sl]-START;st=style('NN' if method=='NN_refined' else method)
            label={'MPC_no_preview':'MPC without preview','NN_refined':'NN'}.get(method,method)
            axs[0 if half else 1].plot(t,tr['y'][sl,0],label=label,**st)
            if half: axs[1].plot(t,tr['y'][sl,1],label=label,**st)
        if half:
            panel(axs[0],'a','Body heave acceleration')
            panel(axs[1],'b','Body pitch acceleration')
            axs[0].set_ylabel(r'Acceleration [m/s$^2$]')
            axs[1].set_ylabel(r'Acceleration [rad/s$^2$]')
            for ax in axs: ax.legend(loc='upper right',fontsize=6.5,ncol=1)
        else:
            axs[0].plot(t,tr['road'][sl]*1000,color='black',lw=1)
            panel(axs[0],'a','Measured road input');axs[0].set_ylabel('Elevation [mm]')
            panel(axs[1],'b','Body acceleration');axs[1].set_ylabel(r'Acceleration [m/s$^2$]')
            axs[1].legend(loc='upper left',fontsize=6.5,ncol=1)
        axs[-1].set_xlabel('Time on evaluation segment [s]');axs[-1].set_xlim(0,6)
        # Reserve headroom for legends so no measured peaks are obscured.
        for ax in ([axs[0], axs[1]] if half else [axs[1]]):
            low,high=ax.get_ylim();ax.set_ylim(low,high+.50*(high-low))
        finish(fig)
        for ext in ['png','pdf','svg']: fig.savefig(OUT/f'{model}_measured_road.{ext}',bbox_inches=None)
        plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true');parser.add_argument('--figures',action='store_true')
    args=parser.parse_args()
    if args.check: print(json.dumps(audit(),indent=2));return
    if args.figures: figures();return
    (OUT/'traces').mkdir(parents=True,exist_ok=True)
    rr=roads();rows=[]
    protocol=dict(dt_s=core.DT,horizon=N,speed_m_s=10,evaluation_s=[2,8],
        segment_count=8,segment_length_m=60,initialization='Static equilibrium at the initial wheel road heights',
        quarter_mass_kg=300,half_body_mass_kg=hc.MS,half_pitch_inertia_kg_m2=hc.INERTIA,
        wheelbase_m=hc.LENGTH,half_axle_unsprung_mass_kg=core.MU,
        half_weights=hc.WEIGHTS.tolist(),half_output_scales=[1,1/hc.ARM,.08,.08,hc.STATIC_LOAD,hc.STATIC_LOAD],
        effort_weight=core.EFFORT,force_limit_per_actuator_N=500,
        metrics='Full 6 s evaluation, after 2 s run-in; means across eight segments; reductions computed from group means.',
        selection='First eight consecutive segments; representative chosen by upper-median FFPD heave RMS.',
        half_scope='Predictive control only; no half-car NN, identification, or timing claim.',
        policy_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'results/models/policy.npz',DEFAULT_POLICY]})
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for model in ['quarter','half']:
        for segment,profile in rr:
            for method in METHODS if model=='quarter' else METHODS[:4]:
                tr=rollout(profile,method,half=model=='half')
                np.savez_compressed(OUT/'traces'/f'{model}__{segment}__{method}.npz',**tr)
                rows.append(dict(model=model,segment=segment,method=method,**metrics(tr,model=='half')))
            print(model,segment,'complete',flush=True)
    # Keep canonical sort for exact summary replay.
    rows.sort(key=lambda r:f"{r['model']}__{r['segment']}__{r['method']}")
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'per_segment.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (OUT/'summary.json').write_text(json.dumps(summarize(rows),indent=2)+'\n')
    print(json.dumps(audit(),indent=2));figures()

if __name__=='__main__': main()
