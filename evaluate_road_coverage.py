"""Frozen candidate checks on old regression roads and new disjoint P79 segments."""
import bootstrap
from bootstrap import ROOT
import argparse,csv,gzip,hashlib,json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import core
from mass_controller import MassConditionedPolicy,MassEstimator,DEFAULT_POLICY
from mass_optimization import MASSES,WINDOWS,savecsv
from refine_road_coverage import OUT,ORIGINAL,INITIAL_POLICY
from compare_tanh_output import score
SOURCE_START=1000


def freeze_protocol():
    p=OUT/'evaluation_protocol.json'
    settings=dict(candidate_sha256=hashlib.sha256((OUT/'policy.npz').read_bytes()).hexdigest(),
                  original_sha256=hashlib.sha256(INITIAL_POLICY.read_bytes()).hexdigest(),
                  existing_measured='Eight original segments: regression only, already inspected in earlier work.',
                  new_measured=f'Eight consecutive segments starting {SOURCE_START} m from the first source record; 2 s run-in and 6 s evaluation at 10 m/s; no overlap with earlier source intervals.',
                  new_source_start_m=SOURCE_START,new_source_end_m=SOURCE_START+510,new_segment_count=8,
                  evaluation='All segments retained. Candidate frozen by synthetic action validation before new measured responses are evaluated.',
                  mass='All 96 original heldout road/mass scenarios and fixed/adaptive/true-mass modes; unchanged RLS.',
                  dataset_use='No measured road samples used for training, normalization or checkpoint selection.')
    if p.exists():assert json.loads(p.read_text())==settings
    else:p.write_text(json.dumps(settings,indent=2)+'\n')


def prepare_source(path):
    raw=Path(path).read_bytes()
    assert hashlib.md5(raw).hexdigest()=='d1c74b645b169dcd88ce99d71ffbb43e'
    lines=raw.decode('cp1252').splitlines()
    rows=list(csv.reader(lines,delimiter=';'))
    data=np.array([[float(r[0]),float(r[21])/1000] for r in rows[1:]])
    distance=data[:,0]-data[0,0]
    use=(distance>=SOURCE_START)&(distance<=SOURCE_START+510)
    assert distance[-1]>SOURCE_START+510 and np.all(np.diff(distance)>0)
    coeff=np.polyfit(distance[use]-SOURCE_START,data[use,1],1)
    elevation=data[:,1]-np.polyval(coeff,distance-SOURCE_START)
    t=np.arange(830)*.01
    profiles=np.array([np.interp(SOURCE_START+i*60+10*t,distance,elevation) for i in range(8)])
    np.savez_compressed(OUT/'new_measured_inputs.npz',profiles=profiles,t=t,detrend_coefficients=coeff)
    ids=np.flatnonzero((distance>=SOURCE_START-.1)&(distance<=SOURCE_START+510.1))
    excerpt=('\n'.join([lines[0]]+[lines[i+1] for i in ids])+'\n').encode('cp1252')
    (OUT/'new_measured_source.csv.gz').write_bytes(gzip.compress(excerpt,mtime=0))
    (OUT/'source_provenance.json').write_text(json.dumps(dict(
        source='https://ndownloader.figshare.com/files/41206187',doi='10.11583/DTU.23097002',
        full_source_md5=hashlib.md5(raw).hexdigest(),excerpt_sha256=hashlib.sha256(excerpt).hexdigest(),
        channel='Laser 21 [mm]',distance_origin=float(data[0,0]),source_range_m=[float(distance[ids[0]]),float(distance[ids[-1]])],
        detrend_fit_range_m=[SOURCE_START,SOURCE_START+510],detrend_coefficients=coeff.tolist(),license='CC BY 4.0',
        scope='Additional measured verification after synthetic-only model freeze; no source amplitude rescaling.'),indent=2)+'\n')


def rollout(road,method,policy=None):
    x=np.array([road[0],0.,road[0],0.]);ad,bd,ed=core.discrete()
    qp=core.MPC() if 'MPC' in method else None
    xs,us,raws,ys,kkt=[],[],[],[],[]
    for k in range(800):
        preview=road[k:k+30].copy()
        if method=='MPC_no_preview':preview[:]=road[k]
        if qp:
            u,_=qp.solve(x,preview);raw=u;kkt.append(qp.last['kkt'])
        elif policy:u,raw=policy.action(x,preview,300.)
        else:raw=-core.KS*x[2]-core.CS*x[3];u=float(np.clip(raw,-500,500))
        xs.append(x.copy());us.append(u);raws.append(raw);ys.append(core.outputs(x,u,road[k]))
        x=ad@x+bd*u+ed*road[k]
    return dict(t=np.arange(800)*.01,road=road[:800],x=np.array(xs),u=np.array(us),raw=np.array(raws),y=np.array(ys),kkt=np.array(kkt),mass=np.full(800,300.))


def evaluate():
    candidate=MassConditionedPolicy(OUT/'policy.npz');current=MassConditionedPolicy(INITIAL_POLICY)
    rows=[];traces=OUT/'traces';traces.mkdir(exist_ok=True)
    groups=[('existing',np.load(ROOT/'results/road_halfcar/road_inputs.npz')['profiles'][:,:,0]),('new',np.load(OUT/'new_measured_inputs.npz')['profiles'])]
    if SOURCE_START!=1000:groups.insert(1,('additional_regression',np.load(ROOT/'results/road_coverage/new_measured_inputs.npz')['profiles']))
    for group,profiles in groups:
        for i,road in enumerate(profiles):
            for method,pol in [('FFPD',None),('MPC_no_preview',None),('MPC',None),('NN_original',current),('NN_expanded',candidate)]:
                tr=rollout(road,method,pol)
                np.savez_compressed(traces/f'{group}__{i}__{method}.npz',**tr)
                rows.append(dict(group=group,segment=i,method=method,**core.metrics(tr,start=2)))
            print('measured',group,i,flush=True)
    savecsv(OUT/'measured_results.csv',rows)
    summary={}
    for group,_ in groups:
        summary[group]={}
        for method in ['FFPD','MPC_no_preview','MPC','NN_original','NN_expanded']:
            rr=[r for r in rows if r['group']==group and r['method']==method]
            summary[group][method]={k:float(np.mean([r[k] for r in rr])) for k in ['acc_rms','travel_rms_mm','tire_rms_N','min_contact_ratio','raw_violation_pct']}
        ss=summary[group]
        summary[group]['nn_vs_ffpd_gain_pct']=100*(1-ss['NN_expanded']['acc_rms']/ss['FFPD']['acc_rms'])
    (OUT/'measured_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
    old=np.load(ORIGINAL)
    act=score((old['test_x'][:,0,:-1]-candidate.mean)/candidate.scale,old['test_y'],candidate.w,candidate.b,'linear')
    (OUT/'action_test.json').write_text(json.dumps(act,indent=2)+'\n')
    core.MassRLS=MassEstimator;rows=[]
    protocol=json.loads((ROOT/'results/mass_optimization/protocol.json').read_text())
    for kind,seeds in protocol['heldout_seeds'].items():
        for seed in seeds:
            for mass in MASSES:
                for mode in ['NN_fixed','NN_adaptive','NN_oracle']:
                    tr=core.rollout(kind,seed,mode,duration=24,policy=candidate,
                                    mass_schedule=lambda t:mass if 4<=t<20 else 300.,acc_noise=.05)
                    for window,(a,b) in WINDOWS.items():
                        sl=(tr['t']>=a)&(tr['t']<b);sub={k:v[sl] for k,v in tr.items() if len(v)==len(sl)}
                        rows.append(dict(kind=kind,seed=seed,mass=mass,mode=mode,window=window,**core.metrics(sub)))
            print('mass',kind,seed,flush=True)
    savecsv(OUT/'mass_results.csv',rows)
    gains=[]
    for kind in protocol['heldout_seeds']:
        for mass in MASSES:
            vals={m:float(np.mean([r['acc_rms'] for r in rows if r['kind']==kind and r['mass']==mass and r['window']=='loaded' and r['mode']==m])) for m in ['NN_fixed','NN_adaptive','NN_oracle']}
            gains.append(dict(kind=kind,mass=mass,**vals,gain_pct=100*(1-vals['NN_adaptive']/vals['NN_fixed'])))
    (OUT/'mass_summary.json').write_text(json.dumps(gains,indent=2)+'\n');print(gains,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source');ap.add_argument('--name',default='road_coverage');ap.add_argument('--source-start',type=int,default=1000);args=ap.parse_args()
    OUT=ROOT/'results'/args.name;SOURCE_START=args.source_start
    with threadpool_limits(limits=1):
        freeze_protocol()
        if args.source:prepare_source(args.source)
        evaluate()
