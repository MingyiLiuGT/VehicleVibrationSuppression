"""Generate manuscript numbers and figures exclusively from executed outputs."""
import bootstrap
from bootstrap import ROOT
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT=ROOT/'results'; FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
# Match the original manuscript's Matplotlib figures: white background, boxed
# axes, blue/red/cyan comparisons, and redundant solid/dashed line encoding.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.labelsize':9,'axes.titlesize':9,'axes.titlepad':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
    'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.75,
    'axes.edgecolor':'black','axes.facecolor':'white','figure.facecolor':'white',
    'lines.linewidth':1.35,'lines.markersize':4.5,'xtick.direction':'out',
    'ytick.direction':'out','xtick.major.width':.7,'ytick.major.width':.7,
    'legend.frameon':True,'legend.fancybox':False,'legend.framealpha':.96,
    'legend.edgecolor':'#BDBDBD','legend.borderpad':.4,
    'legend.handlelength':2.4,'figure.dpi':150,'savefig.dpi':600,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'mathtext.fontset':'dejavusans'})
COLORS={'passive':'#777777','FFPD':'#009FA8','MPC_no_preview':'#A024A8',
        'MPC':'#0000FF','NN':'#FF0000','NN_fixed':'#0000FF',
        'NN_adaptive':'#FF0000','NN_oracle':'#009FA8'}
LINES={'FFPD':'-.','MPC_no_preview':':','MPC':'-','NN':'--',
       'NN_fixed':'-','NN_adaptive':'--','NN_oracle':'-.'}
LABELS={'FFPD':'FFPD','MPC_no_preview':'MPC without preview','MPC':'MPC','NN':'NN',
        'NN_fixed':'Fixed mass','NN_adaptive':'Adaptive mass','NN_oracle':'True mass'}
def style(method):return dict(color=COLORS[method],ls=LINES[method])
def panel(ax,letter,text):ax.set_title(f'({letter}) {text}',loc='left')
# Common manuscript geometry: a 2.70-inch plotting area, approximately the
# width of one panel in the former two-column Figure 2.
PANEL_WIDTH_IN = 2.70
FIGURE_SIZE = (3.70, 4.90)
def finish(fig):
    if len(fig.axes) == 2:
        fig.set_size_inches(*FIGURE_SIZE)
        for ax, bottom in zip(fig.axes, (2.88, .50)):
            ax.set_position([.82/3.70, bottom/4.90, PANEL_WIDTH_IN/3.70, 1.65/4.90])
        replacements = {
            'Acceleration RMS change [%]': 'Acceleration RMS\nchange [%]',
            'RMS reduction vs. passive [%]': 'RMS reduction vs.\npassive [%]',
            'Cumulative acceleration RMS [m/s²]': 'Cumulative acceleration\nRMS [m/s²]',
            'Loaded-segment RMS reduction [%]': 'Loaded-segment\nRMS reduction [%]',
        }
        for ax in fig.axes:
            if ax.get_ylabel() in replacements: ax.set_ylabel(replacements[ax.get_ylabel()])
        if not fig.axes[0].get_xlabel() and fig.axes[1].get_xlabel():
            fig.axes[0].set_xlabel(fig.axes[1].get_xlabel())
            fig.axes[0].tick_params(axis='x',labelbottom=True)
        fig.align_ylabels()
    else:
        fig.align_ylabels()
        fig.tight_layout(pad=.65,w_pad=1.6,h_pad=.8)
def read(name):
    rows=list(csv.DictReader((OUT/name).open()))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v)
            except (ValueError,TypeError):pass
    return rows
def mean(rows,key):return float(np.mean([r[key] for r in rows]))
def std(rows,key):return float(np.std([r[key] for r in rows],ddof=1))
def paired(rows,a,b,key='acc_rms'):
    aa={r['seed']:r[key] for r in rows if r['method']==a}
    bb={r['seed']:r[key] for r in rows if r['method']==b}
    vals=np.array([100*(bb[k]-aa[k])/bb[k] for k in sorted(aa)])
    rng=np.random.default_rng(98765)
    boot=rng.choice(vals,(10000,len(vals)),replace=True).mean(axis=1)
    return dict(mean=float(vals.mean()),sd=float(vals.std(ddof=1)),low=float(np.percentile(boot,2.5)),
                high=float(np.percentile(boot,97.5)),worst=float(vals.min()),best=float(vals.max()),n=len(vals))
def savefig(name):
    for ext in ['png','pdf','svg']:
        plt.savefig(FIG/(name+'.'+ext),bbox_inches=None)
    plt.close()

def summarize():
    rows=read('closed_loop.csv'); sw=read('sweeps.csv'); ids=read('identification.csv'); summary={'closed_loop':{},'sweeps':{},'identification':{}}
    for group in dict.fromkeys(r['group'] for r in rows):
        rr=[r for r in rows if r['group']==group]; group_data={}
        for method in dict.fromkeys(r['method'] for r in rr):
            sub=[r for r in rr if r['method']==method]
            keys=[k for k in sub[0] if k not in ['group','seed','method']]
            group_data[method]={k:dict(mean=mean(sub,k),sd=std(sub,k),min=min(r[k] for r in sub),max=max(r[k] for r in sub)) for k in keys}
        group_data['MPC_vs_FFPD']=paired(rr,'MPC','FFPD')
        group_data['MPC_vs_no_preview']=paired(rr,'MPC','MPC_no_preview')
        group_data['NN_vs_MPC']=paired(rr,'NN','MPC')
        summary['closed_loop'][group]=group_data
    for study in ['force','horizon']:
        summary['sweeps'][study]={}
        for value in sorted(set(r['value'] for r in sw if r['study']==study)):
            rr=[r for r in sw if r['study']==study and r['value']==value]
            summary['sweeps'][study][str(int(value))]={m:dict(mean=mean([r for r in rr if r['method']==m],'acc_rms'),sd=std([r for r in rr if r['method']==m],'acc_rms')) for m in set(r['method'] for r in rr)}
            if study=='force': summary['sweeps'][study][str(int(value))]['preview_gain']=paired(rr,'MPC','MPC_no_preview')
    for noise in [0.,.05,.1]:
        summary['identification'][str(noise)]={}
        for window in ['whole','loaded','unloaded']:
            rr=[r for r in ids if r['noise_sd']==noise and r['window']==window]
            val={m:dict(acc_mean=mean([r for r in rr if r['method']==m],'acc_rms'),acc_sd=std([r for r in rr if r['method']==m],'acc_rms')) for m in ['NN_fixed','NN_adaptive','NN_oracle']}
            val['adaptive_gain']=paired(rr,'NN_adaptive','NN_fixed')
            ada=[r for r in rr if r['method']=='NN_adaptive']
            val['tail_mass_error_pct']=mean(ada,'tail_mass_error_pct')
            for key in ['convergence_load_s','convergence_unload_s']:
                v=[r[key] for r in ada if isinstance(r[key],float)]
                val[key]=dict(mean=float(np.mean(v)) if v else None,successes=len(v),total=len(ada))
            summary['identification'][str(noise)][window]=val
    summary['max_kkt']=max(float(np.max(np.load(p)['kkt'])) for p in (OUT/'traces').glob('*MPC*.npz') if np.load(p)['kkt'].size)
    summary['max_applied_violation_pct']=max(r['violation_pct'] for r in rows)
    summary['minimum_contact_ratio']=min(r['min_contact_ratio'] for r in rows)
    summary['network_seed_robustness']={str(seed):mean([r for r in read('network_seed_robustness.csv') if r['network_seed']==seed],'acc_rms') for seed in [11,22,33]}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    return summary,rows,sw,ids

def representative_horizon_roads(sw):
    """Six input-space medoids; selection does not use controller outcomes."""
    from itertools import combinations
    seeds = sorted({int(r['seed']) for r in sw if r['study'] == 'horizon'})
    inputs = []
    for seed in seeds:
        # Same draw order as core.road_profile('sine', seed).
        rng = np.random.default_rng(seed)
        inputs.append([rng.uniform(.01, .025), rng.uniform(.8, 3.)])
    inputs = np.asarray(inputs)
    scaled = (inputs - inputs.min(axis=0)) / np.ptp(inputs, axis=0)
    distances = np.sum((scaled[:, None] - scaled[None, :])**2, axis=2)
    chosen = min(combinations(range(len(seeds)), 6),
                 key=lambda indices: (float(distances[:, indices].min(axis=1).sum()), indices))
    return [dict(seed=seeds[i], amplitude_m=float(inputs[i, 0]),
                 frequency_Hz=float(inputs[i, 1])) for i in chosen]


def figure_sweeps(s, sw):
    from core import DT
    fig, axs = plt.subplots(2, 1, figsize=FIGURE_SIZE)
    cases = representative_horizon_roads(sw)
    records = []
    for case, label, color, ls, marker in zip(cases, ['Road A', 'Road B', 'Road C', 'Road D', 'Road E', 'Road F'],
            ['#009FA8', '#D17800', '#8B35AD', '#218C3C', '#0000FF', '#FF0000'],
            ['-.', ':', '--', '-.', '-', '--'], ['o', 'v', 'D', 'x', 's', '^']):
        rr = sorted([r for r in sw if r['study'] == 'horizon'
                     and int(r['seed']) == case['seed'] and r['method'] == 'MPC'],
                    key=lambda r: r['value'])
        xs = np.array([r['value'] for r in rr]) * DT
        ys = [r['acc_rms'] for r in rr]
        axs[0].plot(xs, ys, color=color, ls=ls, marker=marker, mfc='white',
                    markersize=3.6, label=label)
        records.append(dict(label=label, **case, prediction_steps=[int(r['value']) for r in rr],
                            prediction_horizon_s=xs.tolist(), acceleration_rms_m_s2=ys))
    axs[0].axvline(30 * DT, color='#777777', ls=':', lw=.8, zorder=0)
    axs[0].text(.31, .23, r'$N=30$', color='#444444', fontsize=8)
    axs[0].set(xlabel='Prediction horizon [s]', ylabel=r'Acceleration RMS [m/s$^2$]',
               xlim=(0, .62), ylim=(-.015, .52), xticks=[0, .1, .2, .3, .4, .5, .6])
    panel(axs[0], 'a', 'Prediction horizon')
    axs[0].legend(fontsize=6.2, loc='upper right', ncol=2, columnspacing=.8, handlelength=2.2)
    force = s['sweeps']['force']; xx = [int(x) for x in force]
    for m, mark in zip(['FFPD', 'MPC_no_preview', 'MPC'], ['o', 's', '+']):
        axs[1].plot(xx, [force[k][m]['mean'] for k in force], marker=mark,
                    mfc='white', label=LABELS[m], **style(m))
    axs[1].set(xlabel='Force bound [N]', ylabel=r'Acceleration RMS [m/s$^2$]')
    panel(axs[1], 'b', 'Force authority'); axs[1].legend(fontsize=7, loc='upper right')
    finish(fig); savefig('fig2_sweeps')
    report = dict(figure='Figure 2(a)', dt_s=DT, terminology='Prediction horizon [s]',
                  definition='T_p = N T_s; N road samples are held over the same N prediction intervals.',
                  selection='Six medoids minimizing summed squared distance in amplitude-frequency space, with each coordinate min-max scaled over the eight original sinusoidal roads. Controller RMS and improvement are not used for selection; ties use ascending road IDs.',
                  cases=records, complete_eight_road_summary=s['sweeps']['horizon'],
                  source='results/sweeps.csv', force_panel='Unchanged eight-road means')
    (OUT/'figure2_horizon_selection.json').write_text(json.dumps(report, indent=2)+'\n')


def figures(s,rows,sw,ids):
    figure_sweeps(s, sw)

    fig,axs=plt.subplots(2,1,figsize=FIGURE_SIZE);hist=read('training_history.csv')
    for run,(seed,color,ls) in enumerate(zip([11,22,33],['#0000FF','#FF0000','#009FA8'],['-','--','-.']),1):
        r=[q for q in hist if q['seed']==seed];axs[0].plot([q['epoch'] for q in r],[500*np.sqrt(q['validation_mse']) for q in r],label=f'Run {run}',color=color,ls=ls,lw=1.1)
    axs[0].set(xlabel='Epoch',ylabel='Validation action RMSE [N]',yscale='log');panel(axs[0],'a','Validation histories');axs[0].legend(fontsize=7)
    axs[0].grid(which='major',color='#DEDEDE',lw=.45);axs[0].set_axisbelow(True)
    from core import Policy
    d=np.load(OUT/'dataset.npz');p=Policy(OUT/'models/policy.npz');target=500*d['test_y'];pred=np.clip(p.predict(d['test_x']),-500,500)
    axs[1].plot([-500,500],[-500,500],'k--',lw=.85,label='Identity')
    axs[1].scatter(target[::8],pred[::8],s=5,alpha=.5,color=COLORS['MPC'],linewidths=0,rasterized=True,label='Test pairs')
    axs[1].set(xlabel='MPC label [N]',ylabel='Applied NN action [N]',xlim=(-550,550),ylim=(-550,550),xticks=[-500,0,500],yticks=[-500,0,500]);panel(axs[1],'b','Held-out action pairs');axs[1].legend(fontsize=7,loc='upper left')
    finish(fig);savefig('fig3_surrogate')

    # Figure 5 isolates road excitation and vibration attenuation. Full-window
    # force/travel metrics and original histories remain in the reproduction data;
    # the manuscript explicitly distinguishes acceleration from state recovery.
    fig,axs=plt.subplots(2,1,figsize=FIGURE_SIZE,sharex=True)
    reference=np.load(OUT/'traces'/'bump_4300_MPC.npz')
    axs[0].plot(reference['t'],reference['road']*1000,color='black',lw=1.15)
    for m in ['FFPD','MPC','NN']:
        tr=np.load(OUT/'traces'/f'bump_4300_{m}.npz')
        np.testing.assert_array_equal(tr['road'],reference['road'])
        axs[1].plot(tr['t'],tr['y'][:,0],label=m,**style(m))
    for ax in axs:
        ax.axhline(0,color='#555555',ls=':',lw=.65,zorder=0);ax.set_xlim(0,3)
    axs[1].legend(ncol=1,loc='upper right',fontsize=7)
    axs[0].set_ylabel('Road elevation [mm]')
    axs[1].set(ylabel=r'Acceleration [m/s$^2$]',xlabel='Time [s]')
    for ax,letter,label in zip(axs,'ab',['Road input','Body acceleration']):panel(ax,letter,label)
    finish(fig);savefig('fig4_bump')

if __name__=='__main__':
    import sys
    if '--figure2-only' in sys.argv:
        figure_sweeps(json.loads((OUT/'summary.json').read_text()), read('sweeps.csv'))
    elif '--figures-only' in sys.argv:
        figures(json.loads((OUT/'summary.json').read_text()),read('closed_loop.csv'),read('sweeps.csv'),read('identification.csv'))
    else:
        figures(*summarize())
