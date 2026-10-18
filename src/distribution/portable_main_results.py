"""Recompute panel descriptions and reaggregate selected saved draws.

Frozen summary extraction is separately labelled and never called refitting.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd

def write(d,p):p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False,float_format='%.15g')
def wq(x,w,q):
    """Merged-tie weighted ECDF quantile with an exact-hit midpoint rule."""
    z=pd.DataFrame({'x':np.asarray(x,dtype=float),'w':np.asarray(w,dtype=float)})
    z=z[np.isfinite(z.x)&np.isfinite(z.w)&(z.w>0)].groupby('x',sort=True).w.sum()
    if z.empty:return float('nan')
    support=z.index.to_numpy(dtype=float);cumulative=np.cumsum(z.to_numpy(dtype=float));total=float(cumulative[-1]);probability=cumulative/total;q=float(q)
    if q<=0:return float(support[0])
    if q>=1:return float(support[-1])
    idx=min(int(np.searchsorted(probability,q,side='left')),len(support)-1)
    hit=idx if abs(float(probability[idx])-q)<=1e-12 else idx-1 if idx>0 and abs(float(probability[idx-1])-q)<=1e-12 else None
    if hit is not None and hit<len(support)-1:return float((support[hit]+support[hit+1])/2.0)
    return float(support[idx])
def iq(x):return float(np.quantile(x,.75)-np.quantile(x,.25))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    pkg=Path(a.package).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    frozen=pkg/'data/reference/historical'
    panel=pd.read_parquet(frozen/'09b_charging_activity_exposure/station_boundary_exposure_panel.parquet')
    unit=pd.read_csv(pkg/'data/derived/common/unit_level_signed.csv',float_precision='round_trip',dtype={'station_id':'string','month':'string'},low_memory=False)
    rows=[];dist=[];actions=[];pair=[];weights=[];ext=[]
    for (system,month,boundary),roster in unit.groupby(['system','month','boundary'],sort=True):
        g=roster[roster.A].copy()
        h=g[g.H];r=g[g.R];c=g[g.C]
        assert len(r)==int(roster.R.sum()) and len(h)==int(roster.H.sum()) and len(c)==int(roster.C.sum())
        rows.append({'system':system,'month':month,'boundary':boundary,'roster_n':len(roster),'A':len(g),'R':len(r),'H':len(h),'C':len(c),'R_C':int(c.R.sum()),'H_C':int(c.H.sum()),'H_over_R':len(h)/len(r) if len(r) else np.nan,'absD_gt090_R':int((r.D30.abs()>.9).sum()),'absD_gt090_H':int((h.D30.abs()>.9).sum())})
        for pop,z in [('A',g),('R',r),('H',h)]:
            for col in ['E_star','S_star','T_star','D30']:
                if len(z):
                    v=z[col].to_numpy();q=np.quantile(v,[.1,.25,.5,.75,.9]);dist.append(dict(system=system,month=month,boundary=boundary,population=pop,quantity=col,n=len(z),p10=q[0],p25=q[1],median=q[2],p75=q[3],p90=q[4]))
        if system=='UrbanEV':ext.append({'month':month,'boundary_minute':int(g.boundary_value.iloc[0]),'A':len(g),'R':len(r),'H':len(h),'H_over_R':len(h)/len(r) if len(r) else np.nan,'H_absD30_over_090':(h.D30.abs()>.9).mean() if len(h) else np.nan})
    write(pd.DataFrame(rows),out/'support_and_direction_by_layer.csv');write(pd.DataFrame(dist),out/'component_and_amount_quantiles.csv');write(pd.DataFrame(ext),out/'external_direction_figure_data.csv')
    for k,g in panel.groupby('layer_order',sort=True):
        r=g[g.denominator_R];e=r.E_star.to_numpy();t=r.T_star.to_numpy();z=t-e
        pair.append({'layer_order':k,'layer_label':g.layer_label.iloc[0],'n_R':len(r),'FIQR':iq(t)-iq(e),'F30':30*(iq(t)-iq(e)),'var_E':np.var(e),'var_T':np.var(t),'var_Z':np.var(z),'cov_E_Z':np.mean((e-e.mean())*(z-z.mean())),'variance_change':np.var(t)-np.var(e)})
        for category in ['service_action','total_outcome']:
            for val,n in r[category].value_counts().items():actions.append({'layer_order':k,'category':category,'value':val,'n':n,'n_R':len(r),'share':n/len(r)})
        for pop in ['A','R']:
            d=g if pop=='A' else r
            for weight in ['ST','TX','KWH']:
                w=d[f'w_{weight}_{pop}'].to_numpy();v=d.B30.abs().to_numpy()
                med=float(np.median(v)) if weight=='ST' else wq(v,w,.5)
                I=lambda x:iq(x) if weight=='ST' else wq(x,w,.75)-wq(x,w,.25)
                for metric,value in [('median_abs_B30',med),('F_IQR',I(d.T_star.to_numpy())-I(d.E_star.to_numpy()))]:weights.append({'layer_order':k,'denominator':pop,'weight':weight,'metric':metric,'selected_estimate':value})
    pair=pd.DataFrame(pair);write(pair,out/'component_pairing_points.csv');write(pd.DataFrame(actions),out/'component_action_outcome.csv');weights=pd.DataFrame(weights);write(weights,out/'weighted_point_estimates.csv')
    old=pd.read_csv(frozen/'robustness_checks/analysis_results/selected_activity_estimates.csv',float_precision='round_trip');old=old[old.scope_type=='layer']
    m=weights.merge(old,on=['layer_order','denominator','weight','metric'],suffixes=('_recomputed','_frozen'),validate='one_to_one')
    err=float(np.max(np.abs(m.selected_estimate_recomputed-m.selected_estimate_frozen)));assert err<1e-8,err
    p=pd.read_csv(frozen/'component_pairing/a2/a2_point_dispersion.csv',float_precision='round_trip');p=p[p.variant=='ORIGINAL']
    pm=pair.merge(p,on='layer_order',suffixes=('_recomputed','_frozen'),validate='one_to_one');perr=float(np.max(np.abs(pm.FIQR_recomputed-pm.FIQR_frozen)));assert perr<1e-10
    saved=pd.read_parquet(frozen/'robustness_checks/analysis_results/covariance_sensitivity_bootstrap_draws.parquet');valid=saved[saved.status=='SUCCESS'];cov=[]
    for col in ['station_marginal_share','shared_segment_marginal_share','independent_residual_marginal_share']:
        lo,hi=np.quantile(valid[col],[.025,.975]);cov.append({'quantity':col,'saved_draws':len(saved),'successful_draws':len(valid),'percentile_lower':lo,'percentile_upper':hi,'operation':'reaggregation of saved bootstrap estimates; no model refit'})
    write(pd.DataFrame(cov),out/'covariance_saved_draw_intervals.csv')
    flow=pd.read_csv(frozen/'01_menu_feasibility/sample_flow.csv')
    total=flow.groupby(['candidate_partition','stage'],as_index=False).n_transactions.sum()
    write(total,out/'sample_flow_totals.csv')
    assert int(total.loc[(total.candidate_partition=='ALL')&(total.stage=='01_raw_file'),'n_transactions'].iloc[0])==8544695
    draw=pd.read_parquet(frozen/'robustness_checks/analysis_results/quantile_sensitivity_bootstrap_draws.parquet')
    if 'selected_estimate' not in draw:draw['selected_estimate']=np.where(draw.weight=='ST',draw.selected_type7_estimate,draw.ecdf_estimate)
    act=[]
    for name,ids in [('JANUARY_LAYER_EQUAL',list(range(1,7))),('JULY_LAYER_EQUAL',list(range(7,11))),('TWO_MONTH_LAYER_EQUAL',list(range(1,11)))]+[(str(k),[k]) for k in range(1,11)]:
        for pop in ['A','R']:
            for metric in ['median_abs_B30','F_IQR']:
                s=draw[draw.layer_order.isin(ids)&(draw.denominator==pop)&(draw.metric==metric)];v=s.pivot_table(index=['draw','layer_order'],columns='weight',values='selected_estimate').groupby(level='draw').mean()
                for w in ['TX','KWH']:
                    delta=(v[w]-v.ST).dropna();lo,hi=np.quantile(delta,[.025,.975]);act.append({'scope':name,'denominator':pop,'metric':metric,'contrast':w+'_MINUS_ST','n_saved_draws':len(delta),'percentile_lower':lo,'percentile_upper':hi,'operation':'reaggregation of original station-bootstrap contrasts; no fresh resampling'})
    write(pd.DataFrame(act),out/'activity_saved_draw_contrasts.csv')
    # Every frozen-summary row retains a precise source and row locator.
    registry=json.loads((pkg/'config/reference_sources.json').read_text(encoding='utf-8'));extract=[]
    for item in registry:
        src=pkg/item['package_path'];d=pd.read_csv(src,float_precision='round_trip');dest=out/'frozen_summary_extraction'/item['id'];write(d,dest.with_suffix('.csv'))
        for i,row in d.iterrows():
            for col,value in row.items():
                if isinstance(value,(int,float,np.integer,np.floating)) and pd.notna(value):extract.append({'source_id':item['id'],'row_zero_based':i,'column':col,'value':value,'operation':'frozen_summary_extraction'})
    write(pd.DataFrame(extract),out/'frozen_numeric_ledger.csv')
    counts=unit.groupby('system')[['A','R','H','C']].sum().astype(int).to_dict('index');assert counts=={'Beijing':{'A':8133,'R':7531,'H':7479,'C':3916},'UrbanEV':{'A':49369,'R':4921,'H':1377,'C':48180}}
    layer_totals=pd.DataFrame(rows).groupby('system')[['A','R','H','C']].sum().astype(int).to_dict('index');assert layer_totals==counts
    assert all(x['n']==int(unit.loc[(unit.system==x['system'])&(unit.month==x['month'])&(unit.boundary==x['boundary']),x['population']].sum()) for x in dist)
    checks={'status':'PASS','primary_counts':counts,'layer_export_counts':layer_totals,'all_quantile_population_sizes_checked':True,'weighted_point_max_abs_difference':err,'pairing_FIQR_max_abs_difference':perr,'covariance_saved_successful_draws':len(valid),'frozen_summary_source_count':len(registry),'frozen_summary_numeric_cells':len(extract),'scope':'Panel estimates recomputed; two saved bootstrap series reaggregated; other numerical summaries extracted with source/row/column locators. No raw reconstruction or model refit.'}
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8');print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
