"""Self-contained replay of the component-resolved price-information assessment.

Derived mode recalculates from packaged rows; --bootstrap selects saved-draw
aggregation or a fresh, identical 2,000-draw station schedule. Raw mode first
rebuilds caches from explicitly supplied Beijing files, then reconstructs prices.
"""
from pathlib import Path
import argparse, importlib.util, json, sys, hashlib, platform
import numpy as np
import pandas as pd

DEFAULT_ROOT=Path(__file__).resolve().parents[1]
KEYS=['scope','layer_order','month','group','weight','configuration']

def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def compare(actual,expected,keys,numeric=None,tolerance=1e-10):
    a=actual.copy();b=expected.copy()
    for k in keys:a[k]=a[k].astype(str);b[k]=b[k].astype(str)
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any(),keys
    z=a.merge(b,on=keys,how='outer',suffixes=('_new','_ref'),indicator=True)
    assert z['_merge'].eq('both').all(),f'Key mismatch: {keys}'
    if numeric is None:numeric=[c for c in a if c not in keys and c in b and pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]
    diffs={}
    for c in numeric:
        x=pd.to_numeric(z[c+'_new']).to_numpy(dtype=float);y=pd.to_numeric(z[c+'_ref']).to_numpy(dtype=float)
        assert np.array_equal(np.isnan(x),np.isnan(y)),f'Missingness differs: {c}'
        mask=np.isfinite(x)&np.isfinite(y)
        value=float(np.max(np.abs(x[mask]-y[mask]))) if mask.any() else 0.
        exact=c.startswith('n_') or c in ['n','draw','layer_order','bootstrap_draws','calibration_n_min','calibration_n_max']
        assert value<=(0 if exact else tolerance),(c,value,tolerance)
        diffs[c]=value
    return {'rows':len(z),'max_abs_difference':max(diffs.values(),default=0),'field_max_differences':diffs,'float_tolerance':tolerance,'counts_exact':True}

def summarize_draws(draws):
    out=[]
    for key,g in draws.groupby(KEYS,sort=False,dropna=False):
        valid=g[g.status.eq('ok')];r=dict(zip(KEYS,key))
        r.update(bootstrap_draws=g.draw.nunique(),n_valid_draws=len(valid),n_invalid_draws=int((~g.status.isin(['ok','empty'])).sum()),n_empty_draws=int(g.status.eq('empty').sum()))
        for col,name in [('mae','point'),('mae','mae'),('me','me'),('p90_abs_error','p90'),('label_disagreement','label_disagreement')]:
            r[name+'_low']=valid[col].quantile(.025);r[name+'_high']=valid[col].quantile(.975)
        out.append(r)
    return pd.DataFrame(out)

def rebuild_b_inputs(source,b):
    p=source.copy(); saved=p[['E','CAL','LOCAL','Y','cal_value','V']].copy()
    d=p.expected_direction
    p['E']=d*(p.formation_after_elec_price-p.formation_before_elec_price)
    p['LOCAL']=d*(p.formation_after_total_price_observed-p.formation_before_total_price_observed)
    p['Z']=p.LOCAL-p.E
    # Use the saved formation contrasts to preserve their original IEEE values
    # after independently checking the endpoint algebra. This also preserves ties.
    endpoint_diffs={c:float(np.max(np.abs(p[c]-saved[c]))) for c in ['E','LOCAL']}
    assert max(endpoint_diffs.values())<=1e-12
    p['E']=source.formation_E_star;p['LOCAL']=source.formation_T_star;p['Z']=p.LOCAL-p.E
    p['Y']=(d*(p.validation_after_total_price_observed-p.validation_before_total_price_observed)).where(p.validation_y_available)
    ydiff=float(np.nanmax(np.abs(p.Y-saved.Y)));assert ydiff<=1e-12
    # Preserve the exact saved observed target after checking the endpoint identity.
    p['Y']=source.Y
    for _,g in p.groupby(['month','boundary_hour'],sort=True):
        for idx,r in g.iterrows():
            other=g.loc[g.station_id.ne(r.station_id),'Z'].dropna().to_numpy()
            p.loc[idx,'cal_value']=b.weighted_quantile_tie(other,np.ones(len(other)),.5)
            p.loc[idx,'cal_n_other_Af']=len(other)
    p['CAL']=p.E+p.cal_value
    p['V']=p.validation_y_available & np.isfinite(p[['E','CAL','LOCAL']]).all(axis=1)
    p['z_flat_group']=np.where(p.Z.abs()<=.03,'absZf_le_0.03','absZf_gt_0.03')
    assert np.array_equal(p.V,saved.V)
    assert np.max(np.abs(p.CAL-saved.CAL))<=1e-12
    assert np.max(np.abs(p.cal_value-saved.cal_value))<=1e-12
    return p,{'endpoint_input_max_differences':endpoint_diffs,'target_endpoint_max_difference':ydiff,'calibration_max_difference':float(np.max(np.abs(p.cal_value-saved.cal_value))),'V_exact':True,'calibration_uses_all_formation_entrants':True,'validation_used_for_inputs':False}

def reproduce_external(root,out):
    adapter=load_module(root/'src/prices/urbanev_adapter.py','replay_external_adapter')
    raw=pd.read_csv(root/'data/derived/external/selected_raw_record_trace.csv.gz',dtype={'station_id':str})
    expected=pd.read_csv(root/'data/reference/external/trace_case_summary.csv',dtype={'station_id':str})
    rows=[];daily=[]
    for r in expected.itertuples():
        x=raw[raw.example_state.eq(r.state)].copy()
        ep=adapter.daily_endpoints(x);pairs=adapter.paired_boundaries(ep,{0:-1,480:1,600:1,720:-1,840:1,1140:-1})
        contrasts=adapter.monthly_contrasts(pairs)
        q=contrasts[contrasts.station_id.eq(r.station_id)&contrasts.month.eq(r.month)&contrasts.boundary_minute.eq(r.boundary_minute)]
        assert len(q)==1
        v=q.iloc[0];rows.append({'state':r.state,'station_id':r.station_id,'month':r.month,'boundary_minute':r.boundary_minute,'paired_days':int(v.paired_days),**{c+'_star':v[c+'_star'] for c in ['E','S','T']}})
        ep.insert(0,'example_state',r.state);daily.append(ep)
    result=pd.DataFrame(rows);check=compare(result,expected,['state','station_id','month','boundary_minute'],['paired_days','E_star','S_star','T_star'],1e-12)
    result.to_csv(out/'external_selected_monthly_recalculated.csv',index=False)
    pd.concat(daily).to_csv(out/'external_selected_daily_recalculated.csv',index=False)
    check['scope']='Selected packaged raw-record chains only; full-panel prevalence remains identified saved evidence.'
    return check

def figures(a_points,b_points,b_ci,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    for cfg in ['original','cd','dw']:
        d=a_points[a_points.config.eq(cfg)&a_points.sample_scope.eq('common_original_H')]
        rate=100*(d.D30_below_minus_0_90.sum()+d.D30_above_plus_0_90.sum())/d.n_units.sum()
        ax.bar(cfg.upper(),rate,color={'original':'#555555','cd':'#0072B2','dw':'#00866A'}[cfg]);ax.text(cfg.upper(),rate+1,f'{rate:.2f}%',ha='center')
    ax.set_ylim(0,45);ax.set_ylabel('Above 0.90 RMB per reference 30 kWh (%)');ax.set_title('Same 7,003 original H units');fig.savefig(out/'core_A_fixed_members.pdf');fig.savefig(out/'core_A_fixed_members.png',dpi=180);plt.close(fig)
    d=b_points[b_points.scope.eq('layer')&b_points.group.eq('all')&b_points.weight.eq('ST')]
    labels=d[['layer_order','month']].drop_duplicates().sort_values(['month','layer_order'])
    fig,ax=plt.subplots(figsize=(7,5.3),layout='constrained')
    for conf,offset,mark in [('E',-.16,'o'),('CAL',0,'s'),('LOCAL',.16,'D')]:
        q=labels.merge(d[d.configuration.eq(conf)],on=['layer_order','month']);ax.plot(q.mae,np.arange(len(q))+offset,mark,ls='',label=conf)
    ax.set_yticks(np.arange(10),[f'{m} layer {l}' for l,m in labels.itertuples(index=False,name=None)]);ax.invert_yaxis();ax.set_xlabel('MAE (RMB/kWh)');ax.legend();ax.grid(axis='x',alpha=.2);fig.savefig(out/'core_B_crosswindow.pdf');fig.savefig(out/'core_B_crosswindow.png',dpi=180);plt.close(fig)

def derived(root,out,bootstrap):
    a=load_module(root/'src/prices/module_a_stats.py','replay_a')
    b=load_module(root/'src/prices/module_b_stats.py','replay_b')
    full=pd.read_csv(root/'data/derived/prices/fullmonth_unit_rows.csv',float_precision='round_trip')
    summaries=[];cov=[]
    for config in ['original','cd','dw']:
        for scope in ['fullsample','common_original_A','common_original_R','common_original_H','common_all_configs']:
            p,c=a.summarize_layer(full,config,scope);summaries.append(p);cov.append(c)
    ap=pd.concat(summaries,ignore_index=True);ac=pd.concat(cov,ignore_index=True)
    ap.to_csv(out/'module_a_layer_summary_recalculated.csv',index=False);ac.to_csv(out/'module_a_covariance_recalculated.csv',index=False)
    audit=a.fixed_member_scope_audit(ap);audit.to_csv(out/'module_a_fixed_scope_audit.csv',index=False)
    refa=pd.read_csv(root/'data/reference/prices/fullmonth_layer_summaries.csv')
    refc=pd.read_csv(root/'data/reference/prices/fullmonth_covariance_decomposition.csv')
    ka=['config','sample_scope','period','month','boundary_hour']
    checks={'module_a_summary':compare(ap,refa,ka),'module_a_covariance':compare(ac,refc,ka)}
    saved=pd.read_csv(root/'data/derived/crosswindow/module_b_row_level_inputs.csv.gz',float_precision='round_trip')
    pairs,bcheck=rebuild_b_inputs(saved,b);checks['module_b_inputs']=bcheck
    point=pd.DataFrame([row for w in ['ST','TX','KWH'] for row in b.layer_metrics(pairs,w)])
    point=pd.concat([point,b.summary_metrics(point)],ignore_index=True)
    refp=pd.read_csv(root/'data/reference/crosswindow/module_b_point_estimates.csv')
    checks['module_b_points']=compare(point,refp,KEYS)
    point.to_csv(out/'module_b_point_estimates_recalculated.csv',index=False)
    cv=b.make_coverage(pairs);checks['coverage']=compare(cv,pd.read_csv(root/'data/reference/crosswindow/module_b_coverage.csv'),['month','layer_order','entry_set'])
    cv.to_csv(out/'module_b_coverage_recalculated.csv',index=False)
    refdraw=pd.read_csv(root/'data/reference/crosswindow/module_b_bootstrap_draws.csv.gz',dtype={'layer_order':str},float_precision='round_trip')
    if bootstrap=='regenerate':
        fast=load_module(root/'src/prices/fast_bootstrap.py','replay_bootstrap')
        draws,ci=fast.bootstrap_metrics(pairs,{'kwh_available':True},['ST','TX','KWH'],2000,20260920)
        checks['bootstrap_draw_comparison']=compare(draws,refdraw,KEYS+['draw'],['n','mae','me','p90_abs_error','label_disagreement'],1e-10)
        draws.to_csv(out/'module_b_bootstrap_draws_regenerated.csv.gz',index=False,compression='gzip')
    else:ci=summarize_draws(refdraw)
    checks['bootstrap_summary']=compare(ci,pd.read_csv(root/'data/reference/crosswindow/module_b_bootstrap_summary.csv',dtype={'layer_order':str}),KEYS)
    ci.to_csv(out/f'module_b_intervals_{bootstrap}.csv',index=False)
    checks['external_trace']=reproduce_external(root,out)
    figures(ap,point,ci,out)
    cal=point[point.scope.eq('overall_equal_10_layers')&point.group.eq('all')&point.weight.eq('ST')&point.configuration.eq('CAL')]
    assert len(cal)==1
    checks.update(status='PASS',mode='derived',bootstrap_mode=bootstrap,bootstrap_draws=2000,seed=20260920,
      formation_entry=len(pairs),common_V=int(pairs.V.sum()),C_f=int(pairs.C_f.sum()),
      station_union=pairs.station_id.nunique(),calibration_and_prices_fixed_during_bootstrap=True,
      shared_station_multiplicities=True,formation_weights_rebuilt_before_conditioning=True,
      overall_ST_MAE_CAL=float(cal.mae.iloc[0]),python_version=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    (out/'reproduction_checks.json').write_text(json.dumps(checks,indent=2,default=int)+'\n',encoding='utf-8')
    return checks

def raw(root,out,rawdir):
    rebuild=load_module(root/'src/prices/raw_reconstruction.py','replay_raw')
    return rebuild.run(root,out,rawdir)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['derived','raw'],required=True)
    p.add_argument('--package-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--raw-dir',type=Path);p.add_argument('--bootstrap',choices=['saved','regenerate'],default='saved')
    args=p.parse_args();root=args.package_root.resolve();out=args.output_dir.resolve()
    if out==root or root in out.parents:raise ValueError('Use a fresh output directory outside the package to protect inputs.')
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory must be empty.')
    out.mkdir(parents=True,exist_ok=True)
    if args.mode=='raw':
        if args.raw_dir is None:p.error('--raw-dir is required in raw mode')
        check=raw(root,out,args.raw_dir.resolve())
    else:check=derived(root,out,args.bootstrap)
    print(json.dumps({k:check[k] for k in ['status','mode'] if k in check}))
if __name__=='__main__':main()
