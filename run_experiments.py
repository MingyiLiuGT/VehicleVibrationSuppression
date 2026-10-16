"""Reproducible revision experiments. Run with --stage all (or individual stages)."""
import bootstrap
from bootstrap import ROOT
import argparse, csv, json, time, platform, subprocess, hashlib, copy
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from sklearn.neural_network import MLPRegressor
from core import *

OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
(OUT/'traces').mkdir(exist_ok=True); (OUT/'models').mkdir(exist_ok=True)

def save_json(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False))

def save_csv(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def dataset():
    all_data={}; scenarios=[]
    for split,n,base in [('train',80,1000),('validation',16,2000),('test',16,3000)]:
        xx=[]; yy=[]; ids=[]; residual=[]
        for ep in range(n):
            seed=base+ep; rng=np.random.default_rng(seed)
            mass=float([240,300,360,420][ep%4])
            # Balance all four masses within each road family; do not confound
            # even/odd road families with even/odd mass-grid indices.
            kind='sine' if (ep//4)%2==0 else 'multisine'
            trace=rollout(kind,seed,mass=mass,duration=4.)
            _,road=road_profile(kind,seed,duration=4.)
            qp=MPC(mass)
            for k,x in enumerate(trace['x']):
                rr=road[k:k+HORIZON]
                xx.append(features(x,rr,mass)); yy.append(trace['u'][k]/500); ids.append(seed)
                # Coverage augmentation belongs to the same scenario split.
                if k%4==0:
                    xp=x+rng.normal(0,[.003,.03,.002,.04])
                    u,_=qp.solve(xp,rr)
                    xx.append(features(xp,rr,mass)); yy.append(u/500); ids.append(seed)
                    residual.append(qp.last['kkt'])
            scenarios.append(dict(split=split,seed=seed,mass=mass,road=kind))
            if ep%10==0: print('dataset',split,ep,'/',n,flush=True)
        all_data[split+'_x']=np.array(xx); all_data[split+'_y']=np.array(yy)
        all_data[split+'_scenario']=np.array(ids)
    np.savez_compressed(OUT/'dataset.npz',**all_data)
    save_json('split_manifest.json',scenarios)

def train():
    d=np.load(OUT/'dataset.npz'); x=d['train_x']; y=d['train_y']
    mean=x.mean(axis=0); scale=x.std(axis=0); scale=np.maximum(scale,1e-6)
    xx=(x-mean)/scale; xv=(d['validation_x']-mean)/scale
    history=[]; selection=[]
    for seed in [11,22,33]:
        model=MLPRegressor(hidden_layer_sizes=(128,128),activation='relu',solver='adam',
                           batch_size=256,learning_rate_init=.001,alpha=1e-5,random_state=seed,
                           max_iter=1,shuffle=True)
        best=np.inf; best_model=None; patience=0; start=time.perf_counter()
        for epoch in range(1,251):
            model.partial_fit(xx,y)
            mse=float(np.mean((np.clip(model.predict(xv),-1,1)-d['validation_y'])**2))
            history.append(dict(seed=seed,epoch=epoch,train_loss=float(model.loss_),validation_mse=mse))
            if mse<best-1e-7:
                best=mse; best_model=copy.deepcopy(model); patience=0; best_epoch=epoch
            else: patience+=1
            if epoch%20==0: print('train',seed,epoch,'validation RMSE N',500*np.sqrt(mse),flush=True)
            if epoch>=80 and patience>=35: break
        params=dict(mean=mean,scale=scale)
        for i,(w,b) in enumerate(zip(best_model.coefs_,best_model.intercepts_)):
            params[f'w{i}']=w; params[f'b{i}']=b
        np.savez(OUT/'models'/f'policy_{seed}.npz',**params)
        selection.append(dict(seed=seed,validation_rmse_N=500*np.sqrt(best),epoch=best_epoch,
                              elapsed_s=time.perf_counter()-start))
    chosen=min(selection,key=lambda q:q['validation_rmse_N'])['seed']
    (OUT/'models'/'policy.npz').write_bytes((OUT/'models'/f'policy_{chosen}.npz').read_bytes())
    save_json('training.json',dict(seeds=selection,selected_seed=chosen,architecture=[35,128,128,1],
              selection_rule='minimum scenario-disjoint validation projected action MSE',
              train_samples=len(y),validation_samples=len(xv),test_samples=len(d['test_y'])))
    save_csv('training_history.csv',history)

def action_metrics(target,raw):
    applied=np.clip(raw,-500,500); error=applied-target; sat=abs(target)>=450
    return dict(samples=len(target),mae_N=float(np.mean(abs(error))),rmse_N=float(np.sqrt(np.mean(error**2))),
        max_error_N=float(np.max(abs(error))),sat_samples=int(sum(sat)),
        sat_mae_N=float(np.mean(abs(error[sat]))) if sum(sat) else None,
        sat_max_error_N=float(np.max(abs(error[sat]))) if sum(sat) else None,
        raw_violation_pct=float(np.mean(abs(raw)>500)*100),applied_violation_pct=float(np.mean(abs(applied)>500)*100))

def evaluate():
    policy=Policy(OUT/'models'/'policy.npz'); d=np.load(OUT/'dataset.npz')
    action=[]
    for seed in [11,22,33]:
        p=Policy(OUT/'models'/f'policy_{seed}.npz')
        action.append(dict(group=f'heldout_seed_{seed}',**action_metrics(500*d['test_y'],p.predict(d['test_x']))))
    rows=[]
    cases=[('in_domain','sine',300.,0.,0),('unseen_sine','unseen_sine',300.,0.,0),
           ('unseen_mass','multisine',330.,0.,0),('bump','bump',300.,0.,0),
           ('preview_noise','multisine',300.,.002,0),('preview_delay','multisine',300.,0.,-2)]
    for ci,(group,kind,mass,noise,shift) in enumerate(cases):
        tg=[]; rr=[]
        for rep in range(8):
            seed=4000+ci*100+rep
            extra=dict(frequency=3.4+.1*rep,amplitude=.018) if kind=='unseen_sine' else {}
            for method in ['passive','FFPD','MPC_no_preview','MPC','NN']:
                trace=rollout(kind,seed,method,mass=mass,policy=policy,preview_noise=noise,
                              preview_shift=shift,**extra)
                met=metrics(trace)
                rows.append(dict(group=group,seed=seed,method=method,**met))
                np.savez_compressed(OUT/'traces'/f'{group}_{seed}_{method}.npz',**trace)
                if method=='NN':
                    _,road=road_profile(kind,seed,**extra)
                    sensor=road+np.random.default_rng(seed+80000).normal(0,noise,len(road))
                    qp=MPC(mass)
                    for k,x in enumerate(trace['x']):
                        idx=np.clip(k+np.arange(HORIZON)+shift,0,len(road)-1)
                        target,_=qp.solve(x,sensor[idx]); tg.append(target); rr.append(trace['raw'][k])
            print('closed loop',group,rep,flush=True)
        action.append(dict(group=group,**action_metrics(np.array(tg),np.array(rr))))
    save_csv('closed_loop.csv',rows); save_json('action_errors.json',action)
    # Additional initialization variability on the same eight in-domain episodes.
    seed_rows=[]
    for net_seed in [11,22,33]:
        p=Policy(OUT/'models'/f'policy_{net_seed}.npz')
        for rep in range(8):
            tr=rollout('sine',4000+rep,'NN',policy=p)
            seed_rows.append(dict(network_seed=net_seed,road_seed=4000+rep,**metrics(tr)))
    save_csv('network_seed_robustness.csv',seed_rows)

def sweeps():
    rows=[]
    for seed in range(5000,5008):
        for horizon in [1,10,30,60]:
            tr=rollout('sine',seed,'MPC',horizon=horizon,duration=6.)
            rows.append(dict(study='horizon',seed=seed,value=horizon,method='MPC',**metrics(tr)))
        for limit in [100.,250.,500.,1000.,2000.]:
            for method in ['FFPD','MPC_no_preview','MPC']:
                tr=rollout('sine',seed,method,limit=limit)
                rows.append(dict(study='force',seed=seed,value=limit,method=method,**metrics(tr,limit)))
        print('sweeps',seed,flush=True)
    save_csv('sweeps.csv',rows)

def identification():
    policy=Policy(OUT/'models'/'policy.npz'); rows=[]
    schedule=lambda t: 300. if t<4 or t>=8 else 420.
    for noise in [0.,.05,.1]:
        for rep in range(8):
            seed=6000+rep
            for method in ['NN_fixed','NN_adaptive','NN_oracle']:
                tr=rollout('multisine',seed,method,duration=12.,policy=policy,mass_schedule=schedule,acc_noise=noise)
                np.savez_compressed(OUT/'traces'/f'id_{noise}_{seed}_{method}.npz',**tr)
                tail=((tr['t']>=7)&(tr['t']<8))|((tr['t']>=11)&(tr['t']<12))
                relative=abs(tr['estimate']-tr['mass'])/tr['mass']*100
                delays=[]
                for change in [4.,8.]:
                    k0=int(change/DT); end=int((change+4)/DT)
                    hit=None
                    for k in range(k0,end-49):
                        if np.all(relative[k:k+50]<5): hit=(k-k0)*DT; break
                    delays.append(hit)
                base=dict(noise_sd=noise,seed=seed,method=method,tail_mass_error_pct=float(np.mean(relative[tail])),
                          convergence_load_s=delays[0],convergence_unload_s=delays[1])
                for start,end,label in [(0,12,'whole'),(4,8,'loaded'),(8,12,'unloaded')]:
                    sub={key:val[(tr['t']>=start)&(tr['t']<end)] for key,val in tr.items() if len(val)==len(tr['t'])}
                    rows.append(dict(**base,window=label,**metrics(sub)))
            print('identification',noise,rep,flush=True)
    save_csv('identification.csv',rows)
    # Retained no-excitation check for mass identification.
    flat=rollout('flat',6999,'NN_adaptive',mass=420.,policy=policy,duration=3.)
    np.savez_compressed(OUT/'traces'/'id_flat.npz',**flat)

def provenance():
    import scipy, sklearn, osqp, matplotlib
    try: cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    except Exception:
        existing=OUT/'provenance.json'
        cpu=json.loads(existing.read_text()).get('cpu',platform.processor()) if existing.exists() else platform.processor()
    save_json('provenance.json',dict(platform=platform.platform(),cpu=cpu,python=platform.python_version(),
      numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,osqp=osqp.__version__,
      matplotlib=matplotlib.__version__,threadpools=threadpool_info(),
      sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
      data='All road inputs are generated synthetic profiles; no measured road or HIL data are used.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='all',choices=['all','dataset','train','evaluate','sweeps','identification'])
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        provenance()
        for name in ['dataset','train','evaluate','sweeps','identification']:
            if args.stage in ('all',name):
                print('START',name,flush=True);globals()[name]();print('DONE',name,flush=True)
