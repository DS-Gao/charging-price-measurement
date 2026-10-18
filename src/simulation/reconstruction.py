"""Known-truth finite-panel transaction-median reconstruction experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

BASE = Path(__file__).resolve().parent
N = 180
BATCH = 8
PROCESSES = ['iid_low', 'iid_high', 'multi_price', 'shared_day', 'selective_support']
LAYOUTS = [(20, 5), (80, 5), (80, 20)]
OFF_E = np.array([0., .025, -.020])
OFF_S = np.array([0., -.140, .120])
PROBS = np.array([[.55, .30, .15], [.35, .45, .20]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                               default=lambda x: x.item() if hasattr(x, 'item') else str(x))+'\n', encoding='utf-8')


def uniform_cdf(x, a):
    return np.clip((np.asarray(x)+a)/(2*a), 0., 1.)


def total_cdf(x, a, b):
    x = np.asarray(x)
    z = (np.maximum(x+a+b, 0.)**2 - np.maximum(x-a+b, 0.)**2
         - np.maximum(x+a-b, 0.)**2 + np.maximum(x-a-b, 0.)**2)/(8*a*b)
    return np.where(x <= -a-b, 0., np.where(x >= a+b, 1., np.clip(z, 0., 1.)))


def mixture_offsets():
    offsets = np.empty((2, 3))
    residuals = []
    for side in range(2):
        for component in range(3):
            off = [OFF_E, OFF_S, OFF_E+OFF_S][component]
            def cdf(x):
                v = (uniform_cdf(x-off, [.015, .020][component]) if component < 2
                     else total_cdf(x-off, .015, .020))
                return float(PROBS[side] @ v)
            offsets[side, component] = brentq(lambda x: cdf(x)-.5, -.5, .5, xtol=1e-14)
            residuals.append(abs(cdf(offsets[side, component])-.5))
    return offsets, max(residuals)


def panel():
    rng = np.random.default_rng(9132610)
    typ = np.repeat(np.arange(6), 30)
    de = np.array([-.020, .003, .009, .011, .050, .200])[typ] + rng.uniform(-.0004, .0004, N)
    ds = np.array([-.070, -.020, 0., .025, .080])[np.arange(N) % 5] + rng.uniform(-.002, .002, N)
    e0 = rng.uniform(.60, .75, N)
    s0 = rng.uniform(.45, .50, N)
    e = np.column_stack([e0, e0+de])
    s = np.column_stack([s0, s0+ds])
    return e, s, typ


def truths(process):
    e, s, typ = panel()
    p = np.stack([e, s, e+s], axis=-1)
    if process == 'multi_price':
        offsets, _ = mixture_offsets()
        p = p + offsets[None, :, :]
    d = p[:, 1, :] - p[:, 0, :]
    return p, d, typ


def categories(x, threshold):
    return np.where(x > threshold, 2, np.where(x < -threshold, 0, 1))


def fiqr(e, t, mask):
    if int(mask.sum()) < 2:
        return np.nan
    eq = np.quantile(e[mask], [.25, .75], method='linear')
    tq = np.quantile(t[mask], [.25, .75], method='linear')
    return float(np.diff(tq)[0]-np.diff(eq)[0])


def generate(rng, process, n_tx, n_days, k):
    e, s, _ = panel()
    shape = (k, N, 2, n_days, n_tx//n_days)
    he, hs = (.06, .08) if process == 'iid_high' else (.015, .020)
    ep = e[None, :, :, None, None] + rng.uniform(-he, he, shape)
    sp = s[None, :, :, None, None] + rng.uniform(-hs, hs, shape)
    if process == 'multi_price':
        u = rng.uniform(size=shape)
        cc = np.zeros(shape, dtype=np.int8)
        for side in range(2):
            cc[:, :, side] = np.searchsorted(np.cumsum(PROBS[side]), u[:, :, side], side='right')
        ep += OFF_E[cc]
        sp += OFF_S[cc]
    if process == 'shared_day':
        day = rng.uniform(-1., 1., (k, N, 2, n_days, 1))
        ep += .04*day
        sp += .06*day
    if np.min(ep) < 0 or np.min(sp) < 0:
        raise ValueError('negative generated component price')
    tp = ep + sp
    m = np.stack([np.median(a.reshape(k, N, 2, n_tx), axis=-1) for a in (ep, sp, tp)], axis=-1)
    dhat = m[:, :, 1, :] - m[:, :, 0, :]
    supported = np.ones((k, N), dtype=bool)
    if process == 'selective_support':
        _, d, _ = truths(process)
        prob = np.where(d[:, 0] > .01, .55, .90)
        endpoint_pass = rng.uniform(size=(k, N, 2)) < prob[None, :, None]
        # Failed endpoints have declared support metadata 8 transactions / 2 days.
        # Their unobserved prices play no part in any reported reconstruction.
        supported = endpoint_pass.all(axis=-1)
    return dhat, supported


def evaluate(dhat, support, truth, rep_start):
    rows = []
    conf_e = np.zeros((3, 3), dtype=np.int64)
    conf_t = conf_e.copy()
    max_closure = 0.
    et, st, tt = truth.T
    ce, ct = categories(et, .01), categories(tt, .03)
    rt = et > .01
    for j in range(len(dhat)):
        a = support[j]
        eh, sh, th = dhat[j].T
        reh = eh > .01
        reft = a & rt
        est = a & reh
        n_a = int(a.sum())
        if n_a == 0:
            raise ValueError('empty supported panel; preserve and investigate run')
        row = {'replicate': rep_start+j, 'n_A': n_a, 'A_coverage': n_a/N,
               'n_true_R': int(reft.sum()), 'n_est_R': int(est.sum()),
               'true_R_prevalence': float(rt[a].mean()), 'estimated_R_prevalence': float(reh[a].mean()),
               'electricity_category_error': float((categories(eh[a], .01) != ce[a]).mean()),
               'total_category_error': float((categories(th[a], .03) != ct[a]).mean()),
               'R_sensitivity': float(reh[reft].mean()) if reft.any() else np.nan,
               'R_precision': float(rt[est].mean()) if est.any() else np.nan}
        near = a & (np.abs(et-.01) <= .003)
        row['near_R_threshold_error'] = float((reh[near] != rt[near]).mean()) if near.any() else np.nan
        far = a & (np.abs(et-.01) > .003)
        row['far_R_threshold_error'] = float((reh[far] != rt[far]).mean()) if far.any() else np.nan
        errors = {'E': eh-et, 'T': th-tt, 'D30': 30*((th-eh)-(tt-et)),
                  'residual': (th-eh-sh)-(tt-et-st)}
        for name, error in errors.items():
            row[name+'_bias'] = float(error[a].mean())
            row[name+'_mae'] = float(np.abs(error[a]).mean())
            row[name+'_mse'] = float((error[a]**2).mean())
        truth_f = fiqr(et, tt, reft)
        est_f = fiqr(eh, th, est)
        price_on_est_f = fiqr(et, tt, est)
        row.update(FIQR_truth_R=truth_f,
                   FIQR_price_error_fixed_true_R=fiqr(eh, th, reft)-truth_f,
                   FIQR_combined_error=est_f-truth_f,
                   FIQR_price_error_given_est_R=est_f-price_on_est_f,
                   FIQR_membership_change=price_on_est_f-truth_f)
        closure = row['FIQR_combined_error']-row['FIQR_price_error_given_est_R']-row['FIQR_membership_change']
        if np.isfinite(closure):
            max_closure = max(max_closure, abs(closure))
        row['FIQR_combined_absolute_error'] = abs(row['FIQR_combined_error'])
        row['FIQR_fixed_R_absolute_error'] = abs(row['FIQR_price_error_fixed_true_R'])
        full_amount = float(np.mean(np.abs(30*tt)))
        a_amount = float(np.mean(np.abs(30*tt[a])))
        obs_amount = float(np.mean(np.abs(30*th[a])))
        row.update(full_true_mean_abs_B30=full_amount, supported_true_mean_abs_B30=a_amount,
                   estimated_mean_abs_B30=obs_amount,
                   amount_reconstruction_error_on_A=obs_amount-a_amount,
                   amount_support_composition_difference=a_amount-full_amount,
                   amount_combined_difference_from_full=obs_amount-full_amount)
        assert abs(row['amount_combined_difference_from_full']-row['amount_reconstruction_error_on_A']
                   -row['amount_support_composition_difference']) < 1e-12
        true_ret = float((tt[reft] > .03).mean()) if reft.any() else np.nan
        est_ret = float((th[est] > .03).mean()) if est.any() else np.nan
        row.update(true_R_retention=true_ret, estimated_R_retention=est_ret, retention_error=est_ret-true_ret)
        np.add.at(conf_e, (ce[a], categories(eh[a], .01)), 1)
        np.add.at(conf_t, (ct[a], categories(th[a], .03)), 1)
        rows.append(row)
    return rows, conf_e, conf_t, max_closure


def run_scenario(idx, phase, reps):
    process = PROCESSES[idx//3]
    n_tx, n_days = LAYOUTS[idx % 3]
    seed = (9132700 if phase == 'pilot' else 9132800) + idx
    rng = np.random.default_rng(seed)
    _, d, _ = truths(process)
    rows, ce, ct, closure = [], np.zeros((3, 3), dtype=np.int64), np.zeros((3, 3), dtype=np.int64), 0.
    for start in range(0, reps, BATCH):
        dh, supported = generate(rng, process, n_tx, n_days, min(BATCH, reps-start))
        rs, a, b, c = evaluate(dh, supported, d, start)
        rows.extend(rs)
        ce += a
        ct += b
        closure = max(closure, c)
    frame = pd.DataFrame(rows)
    frame.insert(0, 'scenario_id', idx)
    frame.insert(1, 'process', process)
    frame.insert(2, 'n_tx_each_endpoint', n_tx)
    frame.insert(3, 'n_days_each_endpoint', n_days)
    frame.insert(4, 'seed', seed)
    return frame, ce, ct, closure


def checks():
    offsets, residual = mixture_offsets()
    cdf_errors = []
    # Independent numerical integration of a uniform CDF over the second uniform.
    for x in np.linspace(-.04, .04, 31):
        exact = float(total_cdf(x, .015, .020))
        points = [v for v in [x-.015, x+.015] if -.02 < v < .02]
        numeric = quad(lambda v: float(uniform_cdf(x-v, .015))/.04,
                       -.02, .02, points=points, epsabs=1e-12)[0]
        cdf_errors.append(abs(exact-numeric))
    total_root_errors = []
    for side in range(2):
        def mix_numeric(x):
            vals = []
            for off in OFF_E+OFF_S:
                points = [v for v in [x-off-.015, x-off+.015] if -.02 < v < .02]
                vals.append(quad(lambda v: float(uniform_cdf(x-off-v, .015))/.04,
                                 -.02, .02, points=points, epsabs=1e-12)[0])
            return float(PROBS[side] @ vals)
        nr = brentq(lambda x: mix_numeric(x)-.5, -.5, .5, xtol=1e-13)
        total_root_errors.append(abs(nr-offsets[side, 2]))
    e, s, _ = panel()
    noiseless = np.stack([e[:, 1]-e[:, 0], s[:, 1]-s[:, 0],
                          (e+s)[:, 1]-(e+s)[:, 0]], axis=-1)
    _, dt, _ = truths('iid_low')
    assert np.array_equal(noiseless, dt)
    assert np.array_equal(categories(noiseless[:, 0], .01), categories(dt[:, 0], .01))
    rng = np.random.default_rng(9132699)
    energy = rng.uniform(5, 35, e.shape)
    fee_error = float(np.max(np.abs((energy*e)/energy-e)))
    assert residual < 1e-11 and max(cdf_errors) < 1e-11 and max(total_root_errors) < 1e-11
    assert fee_error < 1e-12
    return {'mixture_component_total_offsets': offsets.tolist(),
            'mixture_boundary_aggregation_residual_truth': float(np.diff(offsets[:, 2]-offsets[:, 0]-offsets[:, 1])[0]),
            'root_cdf_max_residual': residual, 'analytic_cdf_vs_quadrature_max_error': max(cdf_errors),
            'total_median_vs_independent_quadrature_root_max_error': max(total_root_errors),
            'zero_noise_exact_recovery': True, 'fee_energy_roundtrip_error': fee_error,
            'status': 'PASS'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'formal'], required=True)
    parser.add_argument('--reps', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reps = args.reps or (200 if args.phase == 'pilot' else 2000)
    if reps < 2:
        parser.error('--reps must be at least 2')
    out = args.output.resolve()
    repo = BASE.parents[1]
    if out == repo or repo in out.parents:
        parser.error('--output must be outside the repository')
    if out.exists() and any(out.iterdir()):
        parser.error('--output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    before = datetime.now(timezone.utc).isoformat()
    ck = checks()
    dump(out/'truth_checks.json', ck)
    truth_rows = []
    for process in PROCESSES:
        p, d, typ = truths(process)
        for i in range(N):
            truth_rows.append({'process': process, 'station': i, 'type': int(typ[i]),
                               'E_before': p[i, 0, 0], 'E_after': p[i, 1, 0],
                               'S_before': p[i, 0, 1], 'S_after': p[i, 1, 1],
                               'T_before': p[i, 0, 2], 'T_after': p[i, 1, 2],
                               'E_star': d[i, 0], 'S_star': d[i, 1], 'T_star': d[i, 2],
                               'R': bool(d[i, 0] > .01)})
    pd.DataFrame(truth_rows).to_csv(out/'known_truth_panel.csv', index=False)
    frames, summaries, confusion = [], [], []
    all_closure = 0.
    for idx in range(15):
        tic = time.perf_counter()
        frame, ce, ct, closure = run_scenario(idx, args.phase, reps)
        frames.append(frame)
        all_closure = max(all_closure, closure)
        meta = frame.iloc[0][['scenario_id', 'process', 'n_tx_each_endpoint', 'n_days_each_endpoint', 'seed']].to_dict()
        for metric in frame.columns[6:]:
            x = frame[metric].to_numpy(float)
            valid = np.isfinite(x)
            y = x[valid]
            summaries.append({**meta, 'metric': metric, 'n_replicates': reps,
                              'n_finite': int(valid.sum()), 'n_nonfinite': int((~valid).sum()),
                              'mean': float(y.mean()) if y.size else np.nan,
                              'mcse': float(y.std(ddof=1)/np.sqrt(y.size)) if y.size > 1 else np.nan})
        for kind, conf in [('electricity', ce), ('total', ct)]:
            for a in range(3):
                for b in range(3):
                    confusion.append({**meta, 'component': kind, 'truth_category': a,
                                      'estimate_category': b, 'count': int(conf[a, b])})
        print(f'{args.phase} {idx+1}/15 {meta["process"]} {meta["n_tx_each_endpoint"]}/{meta["n_days_each_endpoint"]}: '
              f'{reps} panels in {time.perf_counter()-tic:.2f}s', flush=True)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(out/'outer_replicates.parquet', index=False)
    pd.DataFrame(summaries).to_csv(out/'performance_summary.csv', index=False)
    pd.DataFrame(confusion).to_csv(out/'confusion_counts.csv', index=False)
    # Prefix reproduction uses same fixed batch and new generator, never reused RNG state.
    prefix, _, _, _ = run_scenario(0, args.phase, min(16, reps))
    pd.testing.assert_frame_equal(prefix.reset_index(drop=True), frames[0].iloc[:len(prefix)].reset_index(drop=True), check_exact=True)
    failcounts = {c: int((~np.isfinite(full[c].to_numpy(float))).sum()) for c in full.columns[6:]}
    assert all_closure < 1e-12
    dump(out/'run_checks.json', {'status': 'COMPLETE', 'started_utc': before,
         'finished_utc': datetime.now(timezone.utc).isoformat(), 'phase': args.phase,
         'n_scenarios': 15, 'replicates_per_scenario': reps, 'n_finite_panels': len(full),
         'latent_stations': N, 'batch_size': BATCH, 'maximum_FIQR_decomposition_residual': all_closure,
         'exact_prefix_reproduction': True, 'nonfinite_counts': failcounts,
         'runtime_seconds': time.perf_counter()-started, 'code_sha256': sha(__file__),
         'design_sha256': sha(BASE/'reconstruction_design.md'),
         'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__})


if __name__ == '__main__':
    main()
