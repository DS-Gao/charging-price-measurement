"""Data-derived external heatmap with explicit small-denominator cues."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.input);months=sorted(d.month.unique());hours=sorted(d.boundary_minute.unique())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,1,figsize=(7.36,7.2));small=[]
    for ax,col,denom,title,cmap in zip(axes,['H_over_R','H_absD30_over_090'],['R','H'],['(a) Direction retained among qualified units','(b) Amount discrepancy among retained units'],['Blues','Oranges']):
        v=d.pivot(index='month',columns='boundary_minute',values=col).reindex(index=months,columns=hours).to_numpy()*100
        n=d.pivot(index='month',columns='boundary_minute',values=denom).reindex(index=months,columns=hours).to_numpy()
        im=ax.imshow(v,vmin=0,vmax=100,cmap=cmap,aspect='auto')
        for i in range(len(months)):
            for j in range(len(hours)):
                tiny=0<n[i,j]<5
                if tiny:
                    ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=(1,1,1,.45),edgecolor='#7a858d',hatch='///',linewidth=.5));small.append({'month':months[i],'boundary_minute':int(hours[j]),'denominator':denom,'n':int(n[i,j])})
                label=f'{v[i,j]:.0f}%\n{denom}={n[i,j]:.0f}'+('*' if tiny else '') if np.isfinite(v[i,j]) else f'NA\n{denom}=0'
                ax.text(j,i,label,ha='center',va='center',fontsize=8.8,color='white' if not tiny and np.isfinite(v[i,j]) and v[i,j]>65 else '#17232b')
        ax.set_yticks(range(len(months)),months);ax.set_xticks(range(len(hours)),[f'{h//60:02}:{h%60:02}' for h in hours]);ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=10);fig.colorbar(im,ax=ax,fraction=.025,pad=.025).set_label('Percent')
    axes[-1].set_xlabel('Reference clock boundary');fig.text(.13,.015,'* Hatched cells: conditioning count below 5. Counts and proportions are retained.',fontsize=8.3)
    fig.subplots_adjust(left=.13,right=.90,bottom=.10,top=.94,hspace=.34)
    for ext in ['pdf','png','svg']:
        kw={'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {}
        fig.savefig(dest/f'Figure_U_direction.{ext}',dpi=180,**kw)
    pd.DataFrame(small).to_csv(dest/'small_denominator_cells.csv',index=False);plt.close(fig);print('Small-denominator cells:',len(small))
if __name__=='__main__':main()
