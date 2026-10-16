"""Independent evidence checks and compact figures for the optimized mass study."""
import bootstrap
from bootstrap import ROOT
import numpy as np,json,csv,hashlib
import core
from mass_controller import MassConditionedPolicy,MassEstimator
from mass_optimization import OUT,WINDOWS
from analyze_results import plt,finish,panel
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=600,bbox_inches=None)
 plt.close(fig)
def plot_mass_trajectory(seed):
 """Full mass history and a fixed two-second response detail (no smoothing)."""
 modes=['NN_fixed','NN_adaptive','NN_oracle']
 tr={m:dict(np.load(OUT/f'mass_heads_final/heldout_traces/sine_{seed}_200_{m}.npz')) for m in modes}
 t=tr['NN_fixed']['t'];detail=(t>=8)&(t<=10)
 fig,axs=plt.subplots(2,1,figsize=(3.7,4.9))
 axs[0].axvspan(8,10,color='#D9D9D9',alpha=.55,zorder=0)
 axs[0].plot(t,tr['NN_fixed']['mass'],'k--',lw=1.1,label='True mass')
 axs[0].plot(t,tr['NN_adaptive']['estimate'],'r-',lw=1.0,label='Online estimate')
 axs[0].legend(ncol=2,fontsize=7,loc='upper center')
 low,high=axs[0].get_ylim();axs[0].set_ylim(low,high+.22*(high-low))
 axs[0].set(ylabel='Sprung mass [kg]',xlabel='Time [s]',xlim=(0,24),xticks=np.arange(0,25,4))
 for change in [4,20]:axs[0].axvline(change,color='gray',ls=':',lw=.8)
 # Draw the reference first so the estimated-mass response remains visible.
 for m,c,ls,lw,label in [('NN_fixed','#0000FF','-',1.15,'Without identification'),('NN_oracle','#009FA8','-.',1.1,'True-mass input'),('NN_adaptive','#FF0000','--',1.15,'With identification')]:
  axs[1].plot(t[detail],tr[m]['y'][detail,0],color=c,ls=ls,lw=lw,label=label)
 handles,labels=axs[1].get_legend_handles_labels()
 axs[1].legend([handles[i] for i in [0,2,1]],[labels[i] for i in [0,2,1]],ncol=1,fontsize=6.5,loc='upper right',framealpha=1)
 axs[1].set(xlabel='Time [s]',ylabel='Acceleration [m/s²]',ylim=(-.15,.26))
 for ax in axs[1:]:ax.set_xlim(8,10);ax.set_xticks(np.arange(8,10.01,.5))
 for ax,letter,title in zip(axs,'ab',['Online mass tracking','Body acceleration: 8–10 s detail']):panel(ax,letter,title)
 finish(fig);save(fig,'mass_trajectory')
def main():
 p=MassConditionedPolicy();old=core.Policy(ROOT/'results/models/policy.npz');z=np.load(OUT/'paired_dense/paired_dataset.npz');xx=z['test_x'];yy=z['test_y']*500
 assert not (set(z['train_scenario'])&set(z['validation_scenario']) or set(z['train_scenario'])&set(z['test_scenario']) or set(z['validation_scenario'])&set(z['test_scenario']))
 test={}
 for name,policy in [('original',old),('optimized',p)]:
  raw=policy.predict(xx.reshape(-1,35)).reshape(yy.shape);u=np.clip(raw,-500,500);err=u-yy;de=(u-u[:,3,None])-(yy-yy[:,3,None]);sat=abs(yy)>=450
  test[name]={'samples':int(yy.size),'mae_N':float(abs(err).mean()),'rmse_N':float(np.sqrt(np.mean(err**2))),'max_error_N':float(abs(err).max()),'sat_mae_N':float(abs(err[sat]).mean()),'sat_samples':int(sat.sum()),'raw_violation_pct':float(np.mean(abs(raw)>500)*100),'mass_delta_rmse_N':float(np.sqrt(np.mean(de[:,np.arange(8)!=3]**2)))}
 checked=0;maxdyn=maxforce=maxreplay=maxpolicy=maxrms=0.;mincontact=1.;maxu=0.;massdyn={m:core.discrete(m) for m in [200,240,300,360,420]}
 for name in ['mass_heads_final','mass_heads_old_estimator','original_reference']:
  policy=old if name=='original_reference' else p
  for path in sorted((OUT/name/'heldout_traces').glob('*.npz')):
   tr=dict(np.load(path));checked+=1;seed=int(path.stem.split('_')[1]);mode='_'.join(path.stem.split('_')[3:]);x=tr['x'];u=tr['u'];mass=tr['mass'];r=tr['road'];y=tr['y'];maxu=max(maxu,float(max(abs(u))));mincontact=min(mincontact,float(np.min(1-y[:,2]/((mass+40)*9.81))))
   acc=(-core.KS*(x[:,0]-x[:,2])-core.CS*(x[:,1]-x[:,3])+u)/mass;maxforce=max(maxforce,float(max(abs(acc-y[:,0]))))
   for m,(a,b,e) in massdyn.items():
    ix=np.flatnonzero(mass[:-1]==m)
    if len(ix):maxdyn=max(maxdyn,float(np.max(abs(x[ix+1]-(x[ix]@a.T+u[ix,None]*b+r[ix,None]*e)))))
   _,road=core.road_profile(path.stem.split('_')[0],seed,24)
   if mode.startswith('NN'):
    sample=np.arange(0,len(u),23);feat=np.stack([np.r_[x[k],road[k:k+30],tr['estimate'][k]] for k in sample]);expected=np.clip(policy.predict(feat),-500,500);maxpolicy=max(maxpolicy,float(max(abs(expected-u[sample]))))
   if mode.endswith('fixed'):assert np.all(tr['estimate']==300)
   if mode.endswith('oracle'):assert np.array_equal(tr['estimate'],mass)
   if mode=='NN_adaptive':
    est=MassEstimator() if name=='mass_heads_final' else MassEstimator(.99,.05);rng=np.random.default_rng(seed+80000);rng.normal(0,0,len(tr['t'])+101)
    for k in range(len(u)):
     maxreplay=max(maxreplay,abs(est.mass-tr['estimate'][k]));est.update(x[k],u[k],y[k,0]+rng.normal(0,.05))
  rows=list(csv.DictReader((OUT/name/'heldout.csv').open()))
  for row in rows:
   path=OUT/name/f'heldout_traces/{row["kind"]}_{row["seed"]}_{int(float(row["mass"]))}_{row["mode"]}.npz';tr=np.load(path);a,b=WINDOWS[row['window']];sl=(tr['t']>=a)&(tr['t']<b);maxrms=max(maxrms,abs(float(np.sqrt(np.mean(tr['y'][sl,0]**2)))-float(row['acc_rms'])))
 assert checked==1056 and maxdyn<1e-12 and maxforce<1e-12 and maxreplay<1e-10 and maxpolicy<1e-8 and maxrms<1e-12 and maxu<=500
 rows=list(csv.DictReader((OUT/'mass_heads_final/heldout.csv').open()));pool=[r for r in rows if r['kind']=='sine' and r['mass']=='200.0' and r['mode']=='NN_fixed' and r['window']=='loaded'];pool.sort(key=lambda r:float(r['acc_rms']));seed=int(pool[len(pool)//2]['seed'])
 stats={}
 for row in rows:
  if row['kind']=='sine' and int(row['seed'])==seed and row['mass']=='200.0':stats.setdefault(row['window'],{})[row['mode']]={k:float(row[k]) for k in ['acc_rms','acc_peak','travel_rms_mm','tire_rms_N','force_rms_N','tail_error_pct']}
 groups=json.loads((OUT/'mass_heads_final/heldout_summary.json').read_text());gaincounts={}
 for kind in ['sine','multisine']:
  for mass in [200,240,360,420]:
   group=[r for r in rows if r['kind']==kind and float(r['mass'])==mass and r['window']=='loaded'];f={r['seed']:float(r['acc_rms']) for r in group if r['mode']=='NN_fixed'};a={r['seed']:float(r['acc_rms']) for r in group if r['mode']=='NN_adaptive'};gaincounts[f'{kind}_{mass}']={'improved':sum(a[s]<f[s] for s in f),'unchanged':sum(a[s]==f[s] for s in f),'total':len(f)}
 report=dict(trajectories=checked,max_transition_residual=maxdyn,max_force_balance_residual=maxforce,max_causal_estimator_replay_error_kg=maxreplay,max_nn_action_replay_error_N=maxpolicy,max_window_rms_error=maxrms,maximum_force_N=maxu,minimum_contact_ratio=mincontact,paired_test=test,representative_seed=seed,representative_selection='Upper median by fixed-controller loaded RMS among twelve 200 kg sine roads; adaptive gain not used.',representative=stats,improvement_counts=gaincounts)
 (OUT/'validation.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='representative'},flush=True)
 import sys
 if '--check-only' in sys.argv:return
 plot_response_figures(seed, rows)

def plot_response_figures(seed, rows):
 # Response figures preserve the manuscript's white/boxed blue-red-cyan style.
 tr={m:dict(np.load(OUT/f'mass_heads_final/heldout_traces/sine_{seed}_200_{m}.npz')) for m in ['NN_fixed','NN_adaptive','NN_oracle']};t=tr['NN_fixed']['t'];colors=['#0000FF','#FF0000','#009FA8'];styles=['-','--','-.'];labels=['Without identification','With identification','True-mass input']
 plot_mass_trajectory(seed)
 fig,axs=plt.subplots(2,1,figsize=(3.7,4.9));sl=(t>=4)&(t<20)
 for m,c,l,label in zip(tr,colors,styles,labels):a=tr[m]['y'][sl,0];axs[0].plot(10*(t[sl]-4),np.sqrt(np.cumsum(a*a)/np.arange(1,len(a)+1)),color=c,ls=l,label=label)
 axs[0].set(xlabel='Distance after mass change [m]',ylabel='Cumulative acceleration RMS [m/s²]',xlim=(0,160));axs[0].legend(fontsize=6)
 masses=[200,240,360,420];positions=np.arange(len(masses));width=.32
 # Use every road and the same ratio of group means as Table 5.
 # Both road families are shown at every mass, including negative gains.
 for kind,color,offset,label in [('sine','#0000FF',-width/2,'Sinusoidal'),('multisine','#FF0000',width/2,'Multisine')]:
  gains=[]
  for mass in masses:
   rr=[r for r in rows if r['kind']==kind and float(r['mass'])==mass and r['window']=='loaded']
   values={mode:[float(r['acc_rms']) for r in rr if r['mode']==mode] for mode in ['NN_fixed','NN_adaptive']}
   assert all(len(v)==12 for v in values.values())
   gains.append(100*(1-np.mean(values['NN_adaptive'])/np.mean(values['NN_fixed'])))
  axs[1].bar(positions+offset,gains,width=width,color=color,label=label,zorder=2)
 axs[1].axhline(0,color='gray',ls=':',lw=.8)
 axs[1].axvline(1.5,color='gray',ls=':',lw=.6)
 axs[1].set(xlabel='Sprung mass after change [kg]',ylabel='Acceleration RMS reduction [%]',xticks=positions,xticklabels=masses,ylim=(-1.8,8.5))
 axs[1].text(.5,7.6,'Mass decrease',ha='center',fontsize=6.5)
 axs[1].text(2.5,7.6,'Mass increase',ha='center',fontsize=6.5)
 axs[1].legend(fontsize=6.5,loc='upper right',bbox_to_anchor=(1,.85))
 panel(axs[0],'a','Post-change route comparison');panel(axs[1],'b','Mass decrease and increase');finish(fig);save(fig,'mass_route')
 plot_timing()
def plot_timing():
 ts=json.loads((OUT/'timing/summary.json').read_text())['methods'];methods=['NN_fixed','MPC_fixed','NN_adaptive','MPC_adaptive'];names=['Fixed\nNN','Fixed\nMPC','Adaptive\nNN','Adaptive\nMPC'];fig,axs=plt.subplots(2,1,figsize=(3.7,4.9));ind=np.arange(4);bottom=np.zeros(4)
 for key,label,col in [('preview_ms','Preview','#777777'),('id_ms','Identification','#FF0000'),('model_ms','Model update','#009FA8'),('control_ms','Control','#0000FF')]:
  v=np.array([ts[m][key]['mean'] for m in methods]);axs[0].bar(ind,v,bottom=bottom,color=col,label=label,width=.6);bottom+=v
 samples=list(csv.DictReader((OUT/'timing/samples.csv').open()))
 values=[np.array([float(r['total_ms']) for r in samples if r['method']==m]) for m in methods]
 means=np.array([v.mean() for v in values]);stds=np.array([v.std(ddof=1) for v in values])
 assert all(len(v)==5000 for v in values)
 assert np.allclose(means,[ts[m]['total_ms']['mean'] for m in methods],rtol=0,atol=1e-12)
 axs[1].bar(ind,means,yerr=stds,width=.6,color='#0000FF',capsize=3,error_kw={'elinewidth':.9,'capthick':.9,'ecolor':'black'},zorder=2)
 for ax in axs:ax.set(xticks=ind,xticklabels=names);ax.tick_params(axis='x',labelrotation=0)
 axs[0].legend(fontsize=6)
 axs[0].set_ylabel('Mean computation [ms]');axs[1].set(ylabel='Software latency [ms]',ylim=(0,max(.58,float(np.max(means+stds))*1.12)));panel(axs[0],'a','Computation by operation');panel(axs[1],'b','Mean ± SD');finish(fig);save(fig,'timing')
 (OUT/'timing/figure4b_statistics.json').write_text(json.dumps({'definition':'Arithmetic mean ± sample standard deviation (ddof=1); all observations retained.','unit':'ms','methods':{m:{'n':len(v),'mean':float(mu),'std':float(sd)} for m,v,mu,sd in zip(methods,values,means,stds)}},indent=2)+'\n')
if __name__=='__main__':
 import sys
 if '--figure6-only' in sys.argv:plot_mass_trajectory(json.loads((OUT/'validation.json').read_text())['representative_seed'])
 elif '--timing-only' in sys.argv:plot_timing()
 elif '--figures-only' in sys.argv:plot_response_figures(json.loads((OUT/'validation.json').read_text())['representative_seed'],list(csv.DictReader((OUT/'mass_heads_final/heldout.csv').open())))
 else:main()
