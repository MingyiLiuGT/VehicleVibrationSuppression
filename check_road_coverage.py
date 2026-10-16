"""Independent physical replay, policy replay and protocol checks for road refinement."""
import bootstrap
from bootstrap import ROOT
import json,csv,hashlib
import numpy as np
from threadpoolctl import threadpool_limits
import core
from mass_controller import MassConditionedPolicy,DEFAULT_POLICY
from refine_road_coverage import INITIAL_POLICY

def main():
 out=ROOT/'results/road_coverage_centered';protocol=json.loads((out/'evaluation_protocol.json').read_text())
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert digest(DEFAULT_POLICY)==digest(out/'policy.npz')==protocol['candidate_sha256']
 assert digest(INITIAL_POLICY)==protocol['original_sha256']
 old=MassConditionedPolicy(INITIAL_POLICY);new=MassConditionedPolicy(out/'policy.npz')
 profiles={'existing':np.load(ROOT/'results/road_halfcar/road_inputs.npz')['profiles'][:,:,0],'additional_regression':np.load(ROOT/'results/road_coverage/new_measured_inputs.npz')['profiles'],'new':np.load(out/'new_measured_inputs.npz')['profiles']}
 rows=list(csv.DictReader((out/'measured_results.csv').open()));assert len(rows)==120
 ad,bd,ed=core.discrete();maxdyn=maxbalance=maxreplay=maxrms=0.
 for r in rows:
  g,i,m=r['group'],int(r['segment']),r['method'];tr=np.load(out/'traces'/f'{g}__{i}__{m}.npz');x,u,y=tr['x'],tr['u'],tr['y'];road=profiles[g][i]
  maxdyn=max(maxdyn,float(np.max(abs(x[1:]-(x[:-1]@ad.T+u[:-1,None]*bd+road[:799,None]*ed)))))
  expected=np.array([core.outputs(xx,uu,rr) for xx,uu,rr in zip(x,u,road)])
  maxbalance=max(maxbalance,float(np.max(abs(expected-y))))
  if m.startswith('NN'):
   p=new if m=='NN_expanded' else old
   for k in range(0,len(x),13):maxreplay=max(maxreplay,abs(p.action(x[k],road[k:k+30],300.)[0]-u[k]))
  maxrms=max(maxrms,abs(float(r['acc_rms'])-float(np.sqrt(np.mean(y[200:,0]**2)))))
  assert np.max(abs(u))<=500 and float(r['min_contact_ratio'])>0
 assert maxdyn<1e-12 and maxbalance<1e-10 and maxreplay<1e-8 and maxrms<1e-12
 rng=np.random.default_rng(1);x=rng.normal(0,.01,(32,35));x[:,-1]=rng.uniform(200,450,32);shifted=x.copy();shifted[:,[0,2]+list(range(4,34))]+=.07
 invariance=float(np.max(abs(new.predict(x)-new.predict(shifted))));assert invariance<1e-8
 data=np.load(ROOT/'results/road_coverage/dataset.npz');assert not set(data['train_scenario'])&set(data['validation_scenario']);assert len(set(data['train_scenario']))==48 and len(set(data['validation_scenario']))==12
 summary=json.loads((out/'measured_summary.json').read_text());assert set(summary)==set(profiles)
 for g in profiles:
  for m in ['FFPD','MPC_no_preview','MPC','NN_original','NN_expanded']:
   vals=[float(r['acc_rms']) for r in rows if r['group']==g and r['method']==m];assert len(vals)==8;assert abs(np.mean(vals)-summary[g][m]['acc_rms'])<1e-12
 report=dict(measured_trajectories=120,measured_segments=24,all_results_retained=True,max_transition_residual=maxdyn,max_output_residual=maxbalance,max_nn_replay_error_N=maxreplay,max_rms_error=maxrms,common_height_invariance_error_N=invariance,policy_sha256=digest(DEFAULT_POLICY),synthetic_train_roads=48,synthetic_validation_roads=12,measured_training=False,all_applied_forces_within_bounds=True,all_tire_contact_positive=True)
 (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':
 with threadpool_limits(limits=1):main()
