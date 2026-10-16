"""Generate the retained framework detail (Figure 1b) and mass sensitivity (Figure 9)."""
import bootstrap
from bootstrap import ROOT
from analyze_results import plt,finish,panel,Rectangle
import json
FIG=ROOT/'results/figures'
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=600,**({'bbox_inches':'tight','pad_inches':.04} if name=='fig1b_framework' else {'bbox_inches':None}))
 plt.close(fig)
def main():
 # Figure 1b: original concept is retained separately in Word.
 fig,ax=plt.subplots(figsize=(7,3.2));ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
 boxes=[(.1,2.7,2.1,1,'State and\nroad preview','black'),(3.1,2.7,3,1,'Mass-conditioned NN\n+ force projection','#0000FF'),(8,2.7,3,1,'Two-DOF\nquarter-car plant','black'),
 (4.0,.65,3.1,1,'Online sprung-mass\nestimator (RLS)','#FF0000'),(8,.65,3,1,'Applied force, state,\nbody acceleration','black'),(3.1,4.2,3,.6,'Offline MPC\ndemonstrations','black')]
 for x,y,w,h,label,col in boxes:
  ax.add_patch(Rectangle((x,y),w,h,facecolor='white',edgecolor=col,lw=1));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
 def arrow(a,b,ls='-'):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=.9,color='black',linestyle=ls))
 for a,b in [((2.2,3.2),(3.1,3.2)),((6.1,3.2),(8,3.2)),((9.5,2.7),(9.5,1.65)),((8,1.15),(7.1,1.15)),((5.55,1.65),(5.55,2.7))]:arrow(a,b)
 arrow((4.6,4.2),(4.6,3.7),'--');ax.text(7.05,3.45,r'$u_k$',ha='center',fontsize=9)
 ax.text(5.7,2.1,r'$\hat m_{s,k-1}$',fontsize=9);ax.text(6,.16,'Estimate updated after actuation; used by the next control command',ha='center',fontsize=8)
 save(fig,'fig1b_framework')
 old=json.loads((ROOT/'results/mass_sensitivity/summary.json').read_text())
 fig,axs=plt.subplots(2,1,figsize=(3.7,4.9))
 for (kind,c,l,name) in [('multisine','#0000FF','-','Multisine'),('sine','#FF0000','--','Sine'),('bump','#009FA8','-.','Bump')]:
  data=old[kind];keys=list(data);xx=[100*float(k) for k in keys]
  axs[0].plot(xx,[data[d]['physical_mass_change_pct']['mean'] for d in keys],color=c,ls=l,label=name)
  axs[1].plot(xx,[data[d]['reduction_versus_passive_pct']['mean'] for d in keys],color=c,ls=l,label=name)
 axs[0].set(xlabel='Mass deviation [%]',ylabel='Acceleration RMS change [%]');axs[0].legend(fontsize=7)
 axs[1].set(xlabel='Mass deviation [%]',ylabel='RMS reduction vs. passive [%]');panel(axs[0],'a','Physical mass effect');panel(axs[1],'b','Retained vibration suppression');finish(fig);save(fig,'fig9_sensitivity')

if __name__=='__main__':main()
