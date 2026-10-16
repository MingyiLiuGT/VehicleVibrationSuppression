"""Expand synthetic road coverage without training on measured test profiles.

Same MPC objective, force limit, input representation, mass heads and RLS.
Checkpoint selection uses separate synthetic action-validation labels only.
"""
import bootstrap
from bootstrap import ROOT
import argparse
import hashlib
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
import core
from mass_controller import MassConditionedPolicy, DEFAULT_POLICY
from mass_optimization import savecsv
from train_paired_mass import GRID
from compare_tanh_output import loss_gradient, score
from train_mass_heads import forward
from train_paired_mass import NOM

OUT = ROOT/'results/road_coverage'
ORIGINAL = ROOT/'results/mass_optimization/paired_dense/paired_dataset.npz'
INITIAL_POLICY = ROOT/'results/road_coverage/initial_policy.npz'


def profile(seed, count=800):
    rng = np.random.default_rng(seed)
    t = np.arange(count+core.HORIZON)*core.DT
    frequency = np.exp(rng.uniform(np.log(.15), np.log(8.), 16))
    phase = rng.uniform(0, 2*np.pi, 16)
    road = np.sum(np.sin(2*np.pi*t[:, None]*frequency+phase)/frequency**.75, axis=1)
    amplitude = float(np.exp(rng.uniform(np.log(.001), np.log(.015))))
    road *= amplitude/np.std(road)
    offset = float(rng.uniform(-.03, .03))
    road += offset
    return road, dict(seed=seed, rms_scale_m=amplitude, offset_m=offset,
                      frequencies_hz=frequency.tolist())


def dataset():
    OUT.mkdir(exist_ok=True)
    protocol = dict(training='Synthetic multisines only; measured profiles are never accessed for training or checkpoint selection.',
                    train_seeds=list(range(21000,21048)), validation_seeds=list(range(22000,22012)),
                    frequencies_hz=[.15,8.], rms_scale_m=[.001,.015], static_offset_m=[-.03,.03],
                    duration_s=8, masses_kg=GRID.tolist(), architecture=[34,128,128,8],
                    source_policy_sha256=hashlib.sha256(INITIAL_POLICY.read_bytes()).hexdigest(),
                    original_data_sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),
                    loss='Same force MSE plus 25 times paired mass-action-difference MSE; balanced original and expanded validation losses.',
                    checkpoint_selection='Minimum sum of original and expanded synthetic validation losses; no measured-road score.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    teachers = [core.MPC(m) for m in GRID]
    arrays, scenarios = {}, []
    for split in ['train','validation']:
        xx, yy, ids = [], [], []
        for i, seed in enumerate(protocol[split+'_seeds']):
            road, meta = profile(seed)
            mass = float(GRID[i % len(GRID)])
            qp = core.MPC(mass)
            ad,bd,ed = core.discrete(mass)
            x = np.array([road[0],0.,road[0],0.])
            rng = np.random.default_rng(seed+100000)
            for k in range(800):
                preview = road[k:k+30]
                u,_ = qp.solve(x,preview)
                if k % 4 == 0:
                    states = [x.copy()]
                    if k % 20 == 0:
                        states.append(x+rng.normal(0,[.0005,.005,.0005,.005]))
                    for state in states:
                        xx.append(np.r_[state,preview])
                        yy.append([q.solve(state,preview)[0]/500 for q in teachers])
                        ids.append(seed)
                x = ad@x+bd*u+ed*road[k]
            scenarios.append(dict(split=split,mass=mass,**meta))
            if i % 8 == 0: print('labels',split,i,flush=True)
        arrays.update({split+'_x':np.array(xx),split+'_y':np.array(yy),split+'_scenario':np.array(ids)})
    np.savez_compressed(OUT/'dataset.npz',**arrays)
    (OUT/'scenarios.json').write_text(json.dumps(scenarios,indent=2))


def center(x):
    x=x.copy();c=x[:,4:].mean(axis=1)
    x[:,0]-=c;x[:,2]-=c;x[:,4:]-=c[:,None]
    return x


def weighted_gradient(x,y,w,b):
    raw,a=forward(x,w,b);ap=np.clip(raw,-1,1)
    weights=np.r_[np.ones(len(x)//2),np.full(len(x)-len(x)//2,4.)][:,None]
    delta=(ap-ap[:,NOM,None])-(y-y[:,NOM,None])
    gd=50*delta/raw.size;gd[:,NOM]-=gd.sum(1)
    g=weights*(2*(raw-y)/raw.size+gd*((raw>-1)&(raw<1)))
    gw,gb=[None]*3,[None]*3
    for j in [2,1,0]:
        gw[j]=a[j].T@g+1e-7*w[j];gb[j]=g.sum(0)
        if j:g=(g@w[j].T)*(a[j]>0)
    return gw+gb


def train(epochs=800,centered=False):
    old, extra = np.load(ORIGINAL), np.load(ROOT/'results/road_coverage/dataset.npz')
    policy = MassConditionedPolicy(INITIAL_POLICY)
    mean,scale = policy.mean,policy.scale
    transform=center if centered else lambda x:x
    orig_raw=transform(old['train_x'][:,0,:-1]);new_raw=transform(extra['train_x'])
    orig_x=(orig_raw-mean)/scale;new_x=(new_raw-mean)/scale
    orig_y,new_y=old['train_y'],extra['train_y']
    orig_vy,new_vy=old['validation_y'],extra['validation_y']
    orig_val=(transform(old['validation_x'][:,0,:-1])-mean)/scale
    new_val=(transform(extra['validation_x'])-mean)/scale
    w,b=[p.copy() for p in policy.w],[p.copy() for p in policy.b]
    params=w+b
    mom,vel=[[np.zeros_like(p) for p in params] for _ in range(2)]
    rng=np.random.default_rng(23001)
    best=np.inf;step=0;history=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        # Balanced minibatches preserve old coverage while learning small inputs.
        order=rng.permutation(len(orig_x));new_order=rng.choice(len(new_x),len(orig_x),replace=True)
        lr=.0001*(.3 if epoch>400 else 1)*(.3 if epoch>650 else 1)
        for k in range(0,len(order),128):
            a,c=order[k:k+128],new_order[k:k+128]
            if centered:
                signs=rng.choice([-1.,1.],size=(len(a)+len(c),1))
                x=(np.r_[orig_raw[a],new_raw[c]]*signs-mean)/scale
                y=np.r_[orig_y[a],new_y[c]]*signs
                grads=weighted_gradient(x,y,w,b)
            else:
                x=np.r_[orig_x[a],new_x[c]];y=np.r_[orig_y[a],new_y[c]]
                _,grads=loss_gradient(x,y,w,b,'linear')
            step+=1
            for p,m,v,g in zip(params,mom,vel,grads):
                m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
                p-=lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        s1=score(orig_val,orig_vy,w,b,'linear')
        s2=score(new_val,new_vy,w,b,'linear')
        objective=s1['loss']+(4 if centered else 1)*s2['loss']
        history.append(dict(epoch=epoch,loss=objective,original_rmse_N=s1['action_rmse_N'],expanded_rmse_N=s2['action_rmse_N']))
        if objective<best:
            best,best_epoch=objective,epoch
            best_w,best_b=[p.copy() for p in w],[p.copy() for p in b]
        if epoch%100==0:print('train',epoch,history[-1],'seconds',round(time.perf_counter()-start,1),flush=True)
    w,b=best_w,best_b
    original_score=score(orig_val,orig_vy,w,b,'linear')
    expanded_score=score(new_val,new_vy,w,b,'linear')
    if centered:
        # Fold the linear centering map into layer 1; runtime architecture is unchanged.
        a=np.r_[np.zeros(4),np.full(30,1/30)]
        c=np.r_[1.,0.,1.,0.,np.ones(30)]
        old_a=w[0]/scale[:,None];new_a=old_a-a[:,None]*(c@old_a)[None,:]
        before=forward(orig_val[:100],w,b)[0]
        w[0]=scale[:,None]*new_a;b[0]=b[0]+mean@(new_a-old_a)
        raw_test=old['validation_x'][:100,0,:-1]
        np.testing.assert_allclose(forward((raw_test-mean)/scale,w,b)[0],before,atol=1e-11)
        assert np.max(abs(c@new_a))<1e-10
    np.savez(OUT/'policy.npz',mean=mean,scale=scale,grid=GRID,
             **{f'w{i}':p for i,p in enumerate(w)},**{f'b{i}':p for i,p in enumerate(b)})
    report=dict(epochs=epochs,best_epoch=best_epoch,seconds=time.perf_counter()-start,
                centered_inputs_folded_into_first_layer=centered,
                validation_original=original_score,
                validation_expanded=expanded_score,
                policy_sha256=hashlib.sha256((OUT/'policy.npz').read_bytes()).hexdigest())
    (OUT/'training.json').write_text(json.dumps(report,indent=2)+'\n')
    savecsv(OUT/'training_history.csv',history)
    print(report,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dataset','train','all'],default='all');ap.add_argument('--epochs',type=int,default=800);ap.add_argument('--centered',action='store_true');args=ap.parse_args()
    if args.stage in ['dataset','all']:
        with threadpool_limits(limits=1):dataset()
    if args.centered:
        OUT=ROOT/'results/road_coverage_centered';OUT.mkdir(exist_ok=True)
        protocol=json.loads((ROOT/'results/road_coverage/protocol.json').read_text())
        protocol.update(input_transform='Subtract preview mean from both displacement states and road elevations; fold this fixed linear map into layer 1 after training.',
                        sign_augmentation='Random joint sign reversal of centered state/preview and all MPC force labels.',
                        validation_objective='Original loss + 4 times expanded low-amplitude loss; no measured profiles for checkpoint selection.',
                        selection_scope='Extension designed after first expanded candidate diagnostics. The 1000–1510 m roads are now regression cases; a new 2000–2510 m set remains unseen until model freeze.')
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with threadpool_limits(limits=1):
        if args.stage in ['train','all']:train(args.epochs,args.centered)
