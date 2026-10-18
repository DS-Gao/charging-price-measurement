"""Explicit raw-data cache construction and fixed-design reconstruction.

No prior author-project cache is read. The packaged original panel is a frozen
membership/reference object; endpoint prices and half-window inputs are rebuilt.
"""
from pathlib import Path
import sys,importlib.util,json,gc
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def run(root,out,rawdir):
    a=load(root/'src/prices/reconstruction_engine.py','raw_a_engine')
    b=load(root/'src/prices/module_b_stats.py','raw_b_engine')
    replay=load(root/'scripts/reproduce_prices.py','raw_replay_helpers')
    config=json.loads((root/'config/price_construction.json').read_text())
    original=pd.read_parquet(root/'data/derived/prices/frozen_original_panel.parquet')
    original=original[original.analysis_scope.eq('PURE_TOU')]
    layers=a.layer_table(config,original)
    cache=out/'raw_cache';cache.mkdir();counters={};cache_rows=[];full_parts=[];half_parts=[]
    for month in ['2025-01','2025-07']:
        path=rawdir/f'orders_{month}_public.parquet'
        if not path.exists():raise FileNotFoundError(path)
        expected=config['inputs'][f'orders_{month}']['sha256']
        actual=a.sha256_file(path)
        if actual!=expected:raise ValueError(f'Raw source hash differs for {month}; do not silently substitute a release.')
        stations=sorted(pq.read_table(path,columns=['station_id']).to_pandas().station_id.dropna().astype(str).unique())
        segs=[tuple([str(x[0]),int(x[1]),int(x[2]),int(x[3])]) for x in config['native_8_segments'][month]]
        targets={(s,x[0]) for s in stations for x in segs};codes={s:i for i,s in enumerate(stations)}
        groups,count,_,_=a.process_order(month,path,targets,codes,segs);data=a.finalize_groups(groups);del groups
        counters[month]=count;native=[]
        for station in stations:
            for segment,*_ in segs:
                row={'month':month,'station_id':station,'segment_id':segment,'candidate_partition':'NATIVE_8'}
                for period,prefix,nmin,dmin in [('fullmonth','',20,5),('formation','formation_',10,3),('validation','validation_',10,3)]:
                    dat=data.get((month,station,segment,period))
                    stats={'E':np.nan,'S':np.nan,'T':np.nan,'n_transactions':0,'n_dates':0,'kwh':0.}
                    if dat is not None:stats=a.summary_from_arrays(dat.E,dat.S,dat.T,dat.tariff_day,dat.kwh,'original')
                    row[prefix+'n_sessions']=int(stats['n_transactions']);row[prefix+'n_tariff_days']=int(stats['n_dates'])
                    support=int(stats['n_transactions'])>=nmin and int(stats['n_dates'])>=dmin
                    row['support_main' if period=='fullmonth' else 'support_'+period]=support
                    if period=='fullmonth':
                        row['median_elec_price']=stats['E'];row['median_service_price']=stats['S'];row['median_total_price_observed']=stats['T']
                    cache_rows.append({'month':month,'station_id':station,'segment_id':segment,'period':period,**stats,'support':support})
                native.append(row)
        native=pd.DataFrame(native);native.to_parquet(cache/f'native_prices_{month}.parquet',index=False)
        # Freeze the original support flags only after raw reconstruction agrees.
        pairs=a.supported_pair_keys(native,layers[layers.month.eq(month)])
        full,half=a.build_unit_rows(pairs,layers,data,native,original)
        full_parts.append(full);half_parts.append(half);del data;gc.collect()
    full=pd.concat(full_parts,ignore_index=True);half=pd.concat(half_parts,ignore_index=True)
    full.to_csv(out/'fullmonth_unit_rows_raw.csv',index=False);half.to_csv(out/'halfmonth_unit_rows_raw.csv',index=False)
    native=pd.concat([pd.read_parquet(cache/f'native_prices_{m}.parquet') for m in ['2025-01','2025-07']],ignore_index=True)
    native.to_parquet(cache/'native_prices.parquet',index=False)
    pd.DataFrame(cache_rows).to_parquet(cache/'endpoint_aggregates.parquet',index=False)
    membership=a.ensure_expected_membership(full,original)
    # Form B from the newly reconstructed original half-window endpoints.
    f=half[half.period.eq('formation')].copy();v=half[half.period.eq('validation')].copy()
    keys=['month','boundary_hour','station_id'];target=v[keys+['original_E_before','original_E_after','original_T_before','original_T_after','original_T_star']]
    p=f.merge(target,on=keys,how='left',suffixes=('','_validation'),validate='one_to_one')
    order={('2025-01',7):1,('2025-01',10):2,('2025-01',13):3,('2025-01',17):4,('2025-01',22):5,('2025-01',23):6,('2025-07',7):7,('2025-07',10):8,('2025-07',22):11,('2025-07',23):12}
    p['layer_order']=[order[(m,int(h))] for m,h in p[['month','boundary_hour']].itertuples(index=False,name=None)]
    p['E']=p.original_E_star;p['LOCAL']=p.original_T_star;p['Z']=p.LOCAL-p.E;p['Y']=p.original_T_star_validation
    p['formation_R']=p.original_R;p['O_v']=p.Y.notna();p['V']=p.O_v
    for _,g in p.groupby(['month','boundary_hour']):
        for idx,r in g.iterrows():
            z=g.loc[g.station_id.ne(r.station_id),'Z'].dropna().to_numpy();p.loc[idx,'cal_value']=b.weighted_quantile_tie(z,np.ones(len(z)),.5);p.loc[idx,'cal_n_other_Af']=len(z)
    p['CAL']=p.E+p.cal_value;p['z_flat_group']=np.where(p.Z.abs()<=.03,'absZf_le_0.03','absZf_gt_0.03')
    for side in ['before','after']:
        p['tx_'+side]=p['original_n_transactions_'+side];p['kwh_'+side]=p['original_kwh_'+side]
    cf=native.groupby(['month','station_id']).agg(n=('segment_id','nunique'),all=('support_formation','all'));cf['C_f']=cf.n.eq(8)&cf['all']
    p=p.merge(cf[['C_f']],on=['month','station_id'],how='left',validate='many_to_one')
    point=pd.DataFrame([r for w in ['ST','TX','KWH'] for r in b.layer_metrics(p,w)]);point=pd.concat([point,b.summary_metrics(point)],ignore_index=True)
    point.to_csv(out/'module_b_point_estimates_raw.csv',index=False);p.to_csv(out/'module_b_inputs_raw.csv.gz',index=False,compression='gzip')
    summary=[];cov=[]
    for cfg in ['original','cd','dw']:
        for scope in ['fullsample','common_original_A','common_original_R','common_original_H','common_all_configs']:
            x,y=a.summarize_layer(full,cfg,scope);summary.append(x);cov.append(y)
    summary=pd.concat(summary);cov=pd.concat(cov)
    summary.to_csv(out/'module_a_layer_summaries_raw.csv',index=False);cov.to_csv(out/'module_a_covariance_raw.csv',index=False)
    checks={'status':'PASS','mode':'raw','scope':'Beijing all-native endpoint caches, original/CD/DW contrasts and B point estimates; no new bootstrap or historical latent-target simulation in raw mode.',
      'raw_counters':counters,'membership':membership,
      'A_summary':replay.compare(summary,pd.read_csv(root/'data/reference/prices/fullmonth_layer_summaries.csv'),['config','sample_scope','period','month','boundary_hour']),
      'B_points':replay.compare(point,pd.read_csv(root/'data/reference/crosswindow/module_b_point_estimates.csv'),replay.KEYS),
      'coverage':replay.compare(b.make_coverage(p),pd.read_csv(root/'data/reference/crosswindow/module_b_coverage.csv'),['month','layer_order','entry_set'])}
    (out/'raw_reconstruction_checks.json').write_text(json.dumps(checks,indent=2,default=int)+'\n')
    return checks
