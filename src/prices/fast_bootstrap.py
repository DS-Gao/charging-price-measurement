"""Vectorized implementation of the fixed-price, shared-station bootstrap.

The sampling schedule and estimands are unchanged; sorting each fixed error
vector once avoids repeating pandas operations and sorting in every draw.
"""
import numpy as np
import pandas as pd

def batch_quantile(values,weights,q=.9,tol=1e-12):
    order=np.argsort(values,kind='stable');v=np.asarray(values)[order]
    unique,starts=np.unique(v,return_index=True)
    mass=np.add.reduceat(weights[:,order],starts,axis=1)
    cum=np.cumsum(mass,axis=1);total=cum[:,-1];target=q*total
    positive=mass>0
    eligible=positive & (cum>=target[:,None]-tol*np.maximum(1,total[:,None]))
    idx=np.argmax(eligible,axis=1);rows=np.arange(len(weights))
    out=unique[idx].copy()
    hit=np.abs(cum[rows,idx]-target)<=tol*np.maximum(1,total)
    later=positive & (np.arange(len(unique))[None,:]>idx[:,None])
    nxt=np.argmax(later,axis=1);use=hit & later.any(axis=1)
    out[use]=(unique[idx[use]]+unique[nxt[use]])/2
    out[total<=0]=np.nan
    return out

def bootstrap_metrics(pairs,activity_info,perspectives,draws_n=2000,seed=20260920):
    stations=np.array(sorted(pairs.station_id.astype(str).unique()))
    sid={s:i for i,s in enumerate(stations)};rng=np.random.default_rng(seed)
    mult=np.stack([np.bincount(rng.choice(len(stations),len(stations),replace=True),minlength=len(stations)) for _ in range(draws_n)]).astype(np.int16)
    frames=[]
    for (month,layer),g in pairs.groupby(['month','layer_order'],sort=True):
        g=g.reset_index(drop=True);v=g.V.to_numpy(bool);m=mult[:,[sid[s] for s in g.station_id]].astype(float)
        errors=g.loc[v,['E','CAL','LOCAL']].to_numpy(float)-g.loc[v,'Y'].to_numpy(float)[:,None]
        labels=lambda x:np.where(x>.03,1,np.where(x<-.03,-1,0))
        wrong=(labels(g.loc[v,['E','CAL','LOCAL']].to_numpy(float))!=labels(g.loc[v,'Y'].to_numpy(float))[:,None]).astype(float)
        for perspective in perspectives:
            if perspective=='ST':w=m[:,v].copy()
            else:
                prefix='tx' if perspective=='TX' else 'kwh'
                bm=m*g[prefix+'_before'].to_numpy(float);am=m*g[prefix+'_after'].to_numpy(float)
                w=(.5*bm/bm.sum(axis=1)[:,None]+.5*am/am.sum(axis=1)[:,None])[:,v]
            totals=w.sum(axis=1);w=np.divide(w,totals[:,None],out=np.zeros_like(w),where=totals[:,None]>0)
            mae=w@np.abs(errors);me=w@errors;disagree=w@wrong
            for j,config in enumerate(['E','CAL','LOCAL']):
                p90=batch_quantile(np.abs(errors[:,j]),w)
                frames.append(pd.DataFrame({'scope':'layer','layer_order':int(layer),'month':str(month),'group':'all','weight':perspective,'configuration':config,'draw':np.arange(draws_n),'n':int(v.sum()),'status':np.where(totals>0,'ok','empty'),'mae':mae[:,j],'me':me[:,j],'p90_abs_error':p90,'label_disagreement':disagree[:,j]}))
            for name,j in [('E_to_LOCAL',0),('CAL_to_LOCAL',1)]:
                frames.append(pd.DataFrame({'scope':'layer','layer_order':int(layer),'month':str(month),'group':'all','weight':perspective,'configuration':name,'draw':np.arange(draws_n),'n':int(v.sum()),'status':np.where(totals>0,'ok','empty'),'mae':mae[:,j]-mae[:,2],'me':np.nan,'p90_abs_error':np.nan,'label_disagreement':np.nan}))
    layers=pd.concat(frames,ignore_index=True);allframes=[layers]
    groupkeys=['group','weight','configuration','draw']
    for month,expected in [('2025-01',6),('2025-07',4),('ALL',10)]:
        part=layers if month=='ALL' else layers.loc[layers.month.eq(month)]
        agg=part.groupby(groupkeys,sort=False).agg(n=('n','sum'),n_layers=('mae','count'),mae=('mae','mean'),me=('me','mean'),p90_abs_error=('p90_abs_error','mean'),label_disagreement=('label_disagreement','mean'),all_ok=('status',lambda x:x.eq('ok').all())).reset_index()
        agg['scope']='overall_equal_10_layers' if month=='ALL' else 'month_equal_layers';agg['month']=month;agg['layer_order']='equal_layers'
        agg['status']=np.where(agg.all_ok & agg.n_layers.eq(expected),'ok','partial');agg=agg.drop(columns='all_ok');allframes.append(agg)
    draws=pd.concat(allframes,ignore_index=True);summaries=[]
    for key,g in draws.groupby(['scope','layer_order','month','group','weight','configuration'],sort=False,dropna=False):
        valid=g.loc[g.status.eq('ok')]
        row=dict(zip(['scope','layer_order','month','group','weight','configuration'],key))
        row.update(bootstrap_draws=draws_n,n_valid_draws=len(valid),n_invalid_draws=int((~g.status.isin(['ok','empty'])).sum()),n_empty_draws=int(g.status.eq('empty').sum()))
        for field,label in [('mae','point'),('mae','mae'),('me','me'),('p90_abs_error','p90'),('label_disagreement','label_disagreement')]:
            row[label+'_low']=valid[field].quantile(.025);row[label+'_high']=valid[field].quantile(.975)
        row['resampling_contract']='prices/calibration fixed; shared station multiplicities; formation activity shares rebuilt then conditioned to V'
        summaries.append(row)
    return draws,pd.DataFrame(summaries)
