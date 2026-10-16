"""Matched development/held-out comparisons for the optimized mass-conditioned NN."""
import bootstrap
from bootstrap import ROOT
import core,numpy as np,json,argparse,hashlib
from mass_optimization import OUT,MASSES,WINDOWS,savecsv,protocol
from threadpoolctl import threadpool_limits
class ExpandedMassRLS(core.MassRLS):
 """Original causal RLS; only projection expanded for the declared 200–450 kg range."""
 def update(self,x,u,measured_acc):
  phi=(-core.KS*(x[0]-x[2])-core.CS*(x[1]-x[3])+u)/300
  if abs(phi)<.05:return self.mass
  gain=self.p*phi/(self.forget+phi*phi*self.p)
  self.theta=float(np.clip(self.theta+gain*(measured_acc-phi*self.theta),300/480,300/180))
  self.p=float(np.clip((self.p-gain*phi*self.p)/self.forget,1e-8,1e4));self.updates+=1
  return self.mass

def run(name,split='development',include_mpc=False):
 p=protocol()
 out=OUT/name
 training=json.loads((out/'training.json').read_text())
 if 'estimator' in training:
  from tune_mass_estimator import TunedRLS
  TunedRLS.forget_value=training['estimator']['forget'];TunedRLS.gate=training['estimator']['gate'];core.MassRLS=TunedRLS
 else:core.MassRLS=ExpandedMassRLS
 if 'grid' in np.load(out/'policy.npz'):
  from mass_controller import MassConditionedPolicy
  pol=MassConditionedPolicy(out/'policy.npz')
 else:pol=core.Policy(out/'policy.npz')
 rows=[];traces=out/f'{split}_traces';traces.mkdir(exist_ok=True)
 settings={'policy_sha256':hashlib.sha256((out/'policy.npz').read_bytes()).hexdigest(),'split':split,'noise':.05,'estimator_projection_kg':[180,480],'estimator':'Original RLS with expanded bounds only','mpc':include_mpc}
 if 'estimator' in training:settings['estimator']=training['estimator']
 settingsfile=out/f'{split}_protocol.json'
 if settingsfile.exists():assert json.loads(settingsfile.read_text())==settings
 else:settingsfile.write_text(json.dumps(settings,indent=2))
 for kind,seeds in p[split+'_seeds'].items():
  for seed in seeds:
   for mass in MASSES:
    vals={}
    methods=['NN_fixed','NN_adaptive','NN_oracle']+(['MPC_fixed','MPC_oracle'] if include_mpc else [])
    for mode in methods:
     path=traces/f'{kind}_{seed}_{int(mass)}_{mode}.npz'
     if path.exists():tr=dict(np.load(path))
     else:
      tr=core.rollout(kind,seed,mode,duration=24,policy=pol,mass_schedule=lambda t:mass if 4<=t<20 else 300.,acc_noise=.05);np.savez_compressed(path,**tr)
     err=abs(tr['estimate']-tr['mass'])/tr['mass']*100
     for win,(a,b) in WINDOWS.items():
      sl=(tr['t']>=a)&(tr['t']<b);sub={k:v[sl] for k,v in tr.items() if len(v)==len(sl)};v=core.metrics(sub)
      rows.append(dict(kind=kind,seed=seed,mass=mass,mode=mode,window=win,tail_error_pct=float(err[(tr['t']>=18)&(tr['t']<20)].mean()),**v))
      if win=='loaded':vals[mode]=v['acc_rms']
    print(name,split,kind,seed,mass,'adaptive/oracle gain',*[round(100*(1-vals[m]/vals['NN_fixed']),3) for m in ['NN_adaptive','NN_oracle']],flush=True)
 savecsv(out/f'{split}.csv',rows);summary=[]
 for kind in p[split+'_seeds']:
  for mass in MASSES:
   for win in WINDOWS:
    rr=[r for r in rows if r['kind']==kind and r['mass']==mass and r['window']==win]
    vals={mode:{k:float(np.mean([r[k] for r in rr if r['mode']==mode])) for k in ['acc_rms','tail_error_pct','travel_rms_mm','tire_rms_N','force_rms_N','min_contact_ratio','acc_peak']} for mode in methods}
    gains={m:100*(1-v['acc_rms']/vals['NN_fixed']['acc_rms']) for m,v in vals.items() if m.startswith('NN')}
    summary.append(dict(kind=kind,mass=mass,window=win,means=vals,gains_pct=gains))
    if win=='loaded':print('MEAN',kind,mass,gains,'fixed/adapt',vals['NN_fixed']['acc_rms'],vals['NN_adaptive']['acc_rms'],flush=True)
 (out/f'{split}_summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='mass_heads_final');ap.add_argument('--split',default='heldout',choices=['development','heldout']);ap.add_argument('--mpc',action='store_true');a=ap.parse_args()
 with threadpool_limits(limits=1):run(a.name,a.split,a.mpc)
