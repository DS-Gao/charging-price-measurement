"""Snapshot adapter. This module does not interpret prices as transaction bills."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Grid:
    starts: tuple[int,...] = (0,480,600,660,720,840,900,1020,1140)
    cadence_minutes: int = 5
    minimum_day_coverage: float = .8


def daily_endpoints(frame: pd.DataFrame, grid: Grid=Grid()):
    """Use paired finite nonnegative E/S records; exclude every duplicate timestamp."""
    x=frame[['station_id','time','e_price','s_price']].copy()
    x['time']=pd.to_datetime(x['time'],errors='coerce')
    x['e_price']=pd.to_numeric(x.e_price,errors='coerce')
    x['s_price']=pd.to_numeric(x.s_price,errors='coerce')
    dup=x.duplicated(['station_id','time'],keep=False)
    valid=(x.time.notna()&~dup&np.isfinite(x.e_price)&np.isfinite(x.s_price)
           &(x.e_price>=0)&(x.s_price>=0)&(x.time.dt.minute%grid.cadence_minutes==0)
           &(x.time.dt.second==0))
    x=x.loc[valid].copy()
    if x.empty: return pd.DataFrame()
    starts=np.asarray(grid.starts)
    assert starts[0]==0 and np.all(np.diff(starts)>0) and starts[-1]<1440
    expected=np.diff(np.r_[starts,1440])//grid.cadence_minutes
    x['minute']=x.time.dt.hour*60+x.time.dt.minute
    x['segment']=np.searchsorted(starts,x.minute,side='right')-1
    x['day']=x.time.dt.normalize()
    x['T']=x.e_price+x.s_price
    x['both_zero']=(x.e_price==0)&(x.s_price==0)
    g=x.groupby(['station_id','day','segment'],sort=True)
    result=g.agg(n=('time','size'),E=('e_price','median'),S=('s_price','median'),
                 T=('T','median'),both_zero_n=('both_zero','sum'),
                 first_time=('time','min'),last_time=('time','max')).reset_index()
    result['expected_slots']=expected[result.segment.to_numpy()]
    result['coverage']=result.n/result.expected_slots
    result['supported']=result.coverage>=grid.minimum_day_coverage
    result['aggregation_residual']=result['T']-result.E-result.S
    return result


def paired_boundaries(endpoints:pd.DataFrame, boundaries:dict[int,int], grid:Grid=Grid()):
    """Return day pairs; midnight joins previous calendar day's final segment."""
    if endpoints.empty: return pd.DataFrame()
    valid=endpoints.loc[endpoints.supported].copy()
    last=len(grid.starts)-1
    outputs=[]
    for boundary,orientation in boundaries.items():
        assert boundary in grid.starts and orientation in (-1,1)
        right_index=grid.starts.index(boundary)
        left_index=(right_index-1) % len(grid.starts)
        left=valid.loc[valid.segment==left_index].copy()
        right=valid.loc[valid.segment==right_index].copy()
        left['boundary_day']=left.day+pd.Timedelta(days=1) if right_index==0 else left.day
        right['boundary_day']=right.day
        pair=left.merge(right,on=['station_id','boundary_day'],suffixes=('_left','_right'))
        if pair.empty: continue
        pair['boundary_minute']=boundary;pair['orientation']=orientation
        pair['month']=pair.boundary_day.dt.strftime('%Y-%m')
        pair['left_segment']=last if right_index==0 else right_index-1
        pair['right_segment']=right_index
        outputs.append(pair)
    return pd.concat(outputs,ignore_index=True) if outputs else pd.DataFrame()


def monthly_contrasts(pairs:pd.DataFrame,min_days:int=10):
    """Difference of endpoint medians over the identical paired-date set per boundary."""
    if pairs.empty: return pd.DataFrame()
    rows=[]
    for key,g in pairs.groupby(['station_id','month','boundary_minute'],sort=True):
        n=g.boundary_day.nunique()
        r={'station_id':key[0],'month':key[1],'boundary_minute':key[2],
           'paired_days':n,'A_supported':n>=min_days}
        d=int(g.orientation.iloc[0])
        for c in ('E','S','T'):
            r[c+'_left']=g[c+'_left'].median();r[c+'_right']=g[c+'_right'].median()
            r[c+'_star']=round(float(d*(r[c+'_right']-r[c+'_left'])),12)
        r['residual_star']=r['T_star']-r['E_star']-r['S_star']
        r['R_qualified']=r['A_supported'] and r['E_star']>.01
        r['signal_status']=('unqualified' if not r['R_qualified'] else
                            'retained' if r['T_star']>.03 else
                            'reversed' if r['T_star']<-.03 else 'neutral')
        r['B30']=30*r['T_star'];r['D30']=30*(r['T_star']-r['E_star'])
        rows.append(r)
    return pd.DataFrame(rows)
