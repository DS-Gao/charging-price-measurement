"""Redraw existing amount/screening results for the concise main text."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--inputdir',required=True)
ap.add_argument('--figuredir',required=True)
ap.add_argument('--checks',required=True)
args=ap.parse_args()
DATA=Path(args.inputdir).resolve()
FIG=Path(args.figuredir).resolve()
FIG.mkdir(parents=True,exist_ok=True)
Path(args.checks).parent.mkdir(parents=True,exist_ok=True)
amt=pd.read_csv(DATA/'amount_summary.csv')
screen=pd.read_csv(DATA/'screening_pooled.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'ps.fonttype':42,'svg.hashsalt':'ae-concise-20260915','axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(10.4,7.2),facecolor='white')
ax=fig.add_axes([.19,.57,.75,.32])
colors={'Beijing':'#35679A','UrbanEV':'#D28239'}
for idx,system in enumerate(colors):
    for j,pop in enumerate(['A','R','H']):
        r=amt[(amt.system==system)&(amt.population==pop)].iloc[0]
        y=2-j+(.19 if idx==0 else -.19)
        v=100*r['abs_gt_0_90_n']/r.n
        ax.barh(y,v,height=.32,color=colors[system])
        ax.text(v-1.4 if v>=20 else v+1.4,y,f'{v:.2f}%',ha='right' if v>=20 else 'left',va='center',fontsize=10,color='white' if v>=20 else '#222222')
ax.set_yticks([2,1,0],['Supported $A$','Qualified $R$','Retained $H$'])
ax.set_xlim(0,105);ax.set_xticks([0,20,40,60,80,100])
ax.set_xlabel('Share with $|D_{30}|>0.90$ RMB (%)',labelpad=7)
ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=c,label=s) for s,c in colors.items()],loc='lower right',bbox_to_anchor=(1,1.02),ncol=2,frameon=False)
fig.text(.04,.956,'a  Amount differences depend on the population described',weight='bold',fontsize=13)

bx=fig.add_axes([.19,.15,.75,.24])
palette=['#35679A','#367B70','#D28239','#E3E6E9']
labels=['Electricity only','Both inputs','Total price only','Neither input']
records=[]
for y,system in [(1,'Beijing'),(0,'UrbanEV')]:
    r=screen[(screen.system==system)&(screen.population=='A')&np.isclose(screen.nominal_fraction,.2)].iloc[0]
    counts=[int(r.E_selected_n-r.intersection_n),int(r.intersection_n),int(r.T_selected_n-r.intersection_n),int(r.stratum_n-r.union_n)]
    assert sum(counts)==int(r.stratum_n)
    left=0
    for n,c in zip(counts,palette):
        pct=100*n/r.stratum_n
        bx.barh(y,pct,left=left,height=.35,color=c,edgecolor='white',linewidth=.7)
        if pct>=5:bx.text(left+pct/2,y,f'{pct:.1f}%',ha='center',va='center',fontsize=10,color='#222222' if c==palette[-1] else 'white')
        left+=pct
    if system=='Beijing':
        detail=f'E: {int(r.E_selected_n):,}; T: {int(r.T_selected_n):,}; symmetric difference: {100*r.symmetric_difference_n/r.stratum_n:.2f}% of A'
    else:
        detail='Both inputs select all 49,369 units: no screening discrimination'
    bx.text(0,y-.29,detail,fontsize=10,va='top')
    records.append({'system':system,'A':int(r.stratum_n),'membership_counts':dict(zip(labels,counts)),'symmetric_difference':int(r.symmetric_difference_n)})
bx.set_yticks([1,0],['Beijing\n$A=8,133$','UrbanEV\n$A=49,369$'])
bx.set_ylim(-.63,1.35);bx.set_xlim(0,100)
bx.set_xticks([0,20,40,60,80,100]);bx.set_xlabel('Share of supported units (%)')
bx.grid(axis='x',alpha=.16);bx.set_axisbelow(True)
fig.text(.04,.454,'b  Nominal 20% screens retain all cutoff ties',weight='bold',fontsize=13)
fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(palette,labels)],loc='lower center',bbox_to_anchor=(.55,.005),ncol=4,frameon=False,fontsize=10)
for ext in ['pdf','svg','png']:
    kw={'dpi':180} if ext=='png' else {}
    if ext=='pdf':kw['metadata']={'CreationDate':None,'ModDate':None,'Creator':'Deterministic Matplotlib code','Title':'Price-input amount and screening consequences'}
    if ext=='svg':kw['metadata']={'Date':None}
    fig.savefig(FIG/f'Figure_core_input_use.{ext}',**kw)
plt.close(fig)
checks={'status':'PASS','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'amount_summary.csv',DATA/'screening_pooled.csv']},'screening':records,'scope':'Redrawing frozen summaries; no change to populations, thresholds, scores or statistical results.'}
Path(args.checks).write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
print('Core input-use figure created from frozen tables.')
