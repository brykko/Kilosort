"""Exploratory: MSR's response to sorting errors applied to real units (default KS4, seed 1, hybrid
excerpt), mirroring the harness's perturbation audit (perturb.py, spec §8.6), with S_i rescored by the
harness's own score.corrected_scores (a unit's S_i depends only on its own counts).

Targets: well-isolated, well-tuned units: eligible, refractory contamination (Hill, plain) <= 5%,
S_i in the top quartile of the sort; up to N_TARGETS drawn at random (seed 0), DRAWS draws each.
Per target unit a:
  contam_other f   add f * n_a spikes drawn from one other random eligible unit (without replacement)
  contam_poisson f add f * n_a homogeneous Poisson spikes
  thin f           delete a random fraction f of a's spikes
  dropout f        delete every spike in one random contiguous block of f of the recording (drift-like)
  dup_in f         add jittered copies (+-2 samples) of a random fraction f of a's spikes to a itself
  merge_<kind>     a merged with b: b random eligible ('random'), preferred direction >= 90 deg away
                   ('diff'), <= 30 deg away ('similar'), or the eligible unit nearest in depth ('near')
  split            a split at random into two halves
  dup_unit         a new unit holding jittered copies (+-2 samples) of 80% of a's spikes (perturb.py)
For each: MSR and dMSR (MSR minus the Poisson baseline at the same spike count) of every unit involved,
and S_i; sum changes for the multi-unit errors (merged unit vs the two originals; halves vs original;
original + copy vs original).
Sort-level (as perturb.py, 5% rate, DRAWS draws, eligibility re-applied by spike counts): merge of
differently tuned units, drop, contaminate (pooled other spikes), thin, split, duplicate units (before
cleaning); changes in S, sum MSR, mean MSR, sum dMSR and mean dMSR over eligible units.
Outputs: C:/data/ks4_tuning/exploratory/msr/perturb_<sort>.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'isolation'))
import common  # noqa: E402
import msr  # noqa: E402
import score  # noqa: E402
from cleaning import load_sort, masked_fraction  # noqa: E402
from comod import unit_estimates  # noqa: E402
from theta_reference import CENTRES, cycle_counts, tuning  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--sort', default='default_s1')
ap.add_argument('--targets', type=int, default=40)
ap.add_argument('--draws', type=int, default=3)
args = ap.parse_args()
FS = 30000
FRACS = (0.05, 0.1, 0.2, 0.3)
THIN = (0.05, 0.1, 0.2, 0.3, 0.5)
rng = np.random.default_rng(0)
t0 = time.time()

sort = common.SORTS[args.sort]
by, iv, T = common.clean_units(sort)
ev = common.evaluation(sort)
units = [int(k) for k in ev['S_i']]
ref = np.load(common.REF)
bounds = ref['cycle_bounds_s'] - common.T0
half = (bounds[:-1] + bounds[1:]) / 2 >= common.DUR / 2
frac_m = masked_fraction(bounds, iv)
dur = np.diff(bounds) * (1 - frac_m)
valid = ref['moving'] & (frac_m <= 0.5) & np.isfinite(ref['theta_ref'])
base = msr.Baseline(common.DUR)


def S_of(trains):
    """Per-unit null-corrected S_i (bits/s) of a list of spike trains."""
    d = {i: t for i, t in enumerate(trains)}
    return score.corrected_scores(ref['theta_ref'], cycle_counts(d, list(d), bounds), dur, valid, half)


def M_of(t):
    m = msr.msr(t, common.DUR)
    return m, m - base(t.size)


# check S_of against the evaluation
chk = S_of([by[c] for c in units[:50]])
assert np.allclose(chk, [ev['S_i'][str(c)] for c in units[:50]], atol=1e-9), 'S_i mismatch'

S0 = np.array([ev['S_i'][str(c)] for c in units])
num, occ = tuning(np.where(valid, ref['theta_ref'], 0.0), cycle_counts(by, units, bounds), dur, valid)
pd = dict(zip(units, np.angle((num / np.maximum(occ, 1e-9)) @ np.exp(1j * CENTRES))))
_, _, ids, xy = load_sort(sort)
depth = dict(zip(ids.tolist(), xy[:, 1]))
con = {c: unit_estimates(by[c], T)[2]['plain'] for c in units}
s_q3 = np.quantile(S0, 0.75)
pool = [c for c in units if con[c] <= 0.05 and ev['S_i'][str(c)] >= s_q3]
targets = [int(c) for c in rng.choice(pool, min(args.targets, len(pool)), replace=False)]
print(f'{len(units)} eligible, {len(pool)} well-isolated top-quartile units, {len(targets)} targets', flush=True)


def jitter_copy(t, f):
    s = t[rng.random(t.size) < f]
    return s + rng.integers(-2, 3, s.size) / FS


def ang(a, b):
    return abs(np.angle(np.exp(1j * (pd[a] - pd[b]))))


rows = []
for a in targets:
    ta = by[a]
    m0, d0 = M_of(ta)
    s0 = ev['S_i'][str(a)]
    rec = {'unit': a, 'n': int(ta.size), 'rate': ta.size / T, 'msr': m0, 'dmsr': d0, 'S_i': s0, 'pert': {}}
    others = [c for c in units if c != a]
    for _ in range(args.draws):
        single = []                           # (label, train)
        for f in FRACS:
            b = rng.choice(others)
            tb = by[b]
            extra = rng.choice(tb, int(f * ta.size), replace=tb.size < f * ta.size)
            single.append((f'contam_other_{f}', np.sort(np.concatenate([ta, extra]))))
            single.append((f'contam_poisson_{f}', np.sort(np.concatenate([ta, rng.uniform(0, common.DUR, int(f * ta.size))]))))
            single.append((f'dup_in_{f}', np.sort(np.concatenate([ta, jitter_copy(ta, f)]))))
        for f in THIN:
            single.append((f'thin_{f}', ta[rng.random(ta.size) >= f]))
        for f in (0.1, 0.3):
            s = rng.uniform(0, common.DUR * (1 - f))
            single.append((f'dropout_{f}', ta[(ta < s) | (ta >= s + f * common.DUR)]))
        S_single = S_of([t for _, t in single])
        for (lab, t), s in zip(single, S_single):
            m, d = M_of(t)
            rec['pert'].setdefault(lab, []).append({'dmsr_change': d - d0, 'msr_change': m - m0, 'dS': float(s - s0)})
        # merges
        kinds = {'random': others,
                 'diff': [c for c in others if ang(a, c) >= np.pi / 2],
                 'similar': [c for c in others if ang(a, c) <= np.pi / 6],
                 'near': sorted(others, key=lambda c: abs(depth[c] - depth[a]))[:3]}
        multi = []
        for k, cand in kinds.items():
            b = int(rng.choice(cand))
            multi.append((f'merge_{k}', [np.sort(np.concatenate([ta, by[b]]))], [a, b]))
        side = rng.random(ta.size) < 0.5
        multi.append(('split', [ta[side], ta[~side]], [a]))
        multi.append(('dup_unit', [ta, np.sort(jitter_copy(ta, 0.8))], [a]))
        for lab, trains, orig in multi:
            s_new = S_of(trains)
            mm = [M_of(t) for t in trains]
            mo = [M_of(by[c]) for c in orig]
            s_orig = [ev['S_i'][str(c)] for c in orig]
            rec['pert'].setdefault(lab, []).append({
                'msr_new': [x[0] for x in mm], 'dmsr_new': [x[1] for x in mm],
                'msr_orig': [x[0] for x in mo], 'dmsr_orig': [x[1] for x in mo],
                'sum_msr_change': sum(x[0] for x in mm) - sum(x[0] for x in mo),
                'sum_dmsr_change': sum(x[1] for x in mm) - sum(x[1] for x in mo),
                'mean_msr_new_minus_max_orig': float(np.mean([x[0] for x in mm]) - max(x[0] for x in mo)),
                'dS_sum': float(s_new.sum() - sum(s_orig)), 'S_new': s_new.tolist(), 'S_orig': s_orig,
                'n_orig': [int(by[c].size) for c in orig]})
    rows.append(rec)
    print(f'unit {a} done ({time.time() - t0:.0f} s)', flush=True)


def summarise(rows):
    out = {}
    labs = rows[0]['pert'].keys()
    for lab in labs:
        e = [x for r in rows for x in r['pert'][lab]]
        if 'dmsr_change' in e[0]:
            per_unit_dm = np.array([np.mean([x['dmsr_change'] for x in r['pert'][lab]]) for r in rows])
            per_unit_ds = np.array([np.mean([x['dS'] for x in r['pert'][lab]]) for r in rows])
            out[lab] = {'median_dmsr_change': float(np.median(per_unit_dm)),
                        'mean_dmsr_change': float(np.mean(per_unit_dm)),
                        'frac_units_dmsr_down': float(np.mean(per_unit_dm < 0)),
                        'median_msr_change': float(np.median([np.mean([x['msr_change'] for x in r['pert'][lab]]) for r in rows])),
                        'median_dS': float(np.median(per_unit_ds)), 'frac_units_dS_down': float(np.mean(per_unit_ds < 0)),
                        'median_rel_dS': float(np.median(per_unit_ds / np.array([r['S_i'] for r in rows])))}
        else:
            g = lambda k: np.array([x[k] for x in e])  # noqa: E731
            dmn = np.array([np.mean(x['dmsr_new']) - max(x['dmsr_orig']) for x in e])
            out[lab] = {'median_sum_msr_change': float(np.median(g('sum_msr_change'))),
                        'median_sum_dmsr_change': float(np.median(g('sum_dmsr_change'))),
                        'frac_sum_dmsr_down': float(np.mean(g('sum_dmsr_change') < 0)),
                        'median_mean_msr_new_minus_max_orig': float(np.median(g('mean_msr_new_minus_max_orig'))),
                        'median_mean_dmsr_new_minus_max_orig': float(np.median(dmn)),
                        'frac_mean_dmsr_new_below_max_orig': float(np.mean(dmn < 0)),
                        'median_dS_sum': float(np.median(g('dS_sum'))), 'frac_dS_sum_down': float(np.mean(g('dS_sum') < 0))}
    return out


res = {'sort': args.sort, 'n_eligible': len(units), 'n_pool': len(pool), 'targets': targets, 'draws': args.draws,
       'summary': summarise(rows), 'units': rows}
print(json.dumps(res['summary'], indent=1), flush=True)

# ---- sort-level audit as perturb.py (5%) ----
RATE = 0.05
n_pert = int(round(RATE * len(units)))
elig0 = set(units)


def sort_level(d):
    """S, sum/mean MSR and dMSR over units of d passing the spike-count criterion."""
    keys = [k for k, t in d.items() if min(np.sum(t < common.DUR / 2), np.sum(t >= common.DUR / 2)) >= 100]
    s = S_of([d[k] for k in keys])
    mm = np.array([M_of(d[k]) for k in keys])
    return {'S': float(s.sum()), 'sum_msr': float(mm[:, 0].sum()), 'mean_msr': float(mm[:, 0].mean()),
            'sum_dmsr': float(mm[:, 1].sum()), 'mean_dmsr': float(mm[:, 1].mean()), 'n': len(keys)}


d0 = {c: by[c] for c in units}
L0 = sort_level(d0)
all_t = np.concatenate([by[c] for c in units])


def p_merge():
    d = dict(d0)
    for a in rng.choice(units, n_pert, replace=False):
        if a not in d:
            continue
        cand = [b for b in units if b != a and b in d and ang(a, b) >= np.pi / 2]
        b = rng.choice(cand)
        d[a] = np.sort(np.concatenate([d[a], d.pop(b)]))
    return d


def p_drop():
    d = dict(d0)
    for a in rng.choice(units, n_pert, replace=False):
        d.pop(a)
    return d


def p_contam():
    return {c: np.sort(np.concatenate([t, rng.choice(all_t, int(RATE * t.size))])) for c, t in d0.items()}


def p_thin():
    return {c: t[rng.random(t.size) >= RATE] for c, t in d0.items()}


def p_split():
    d, new = dict(d0), max(d0) + 1
    for a in rng.choice(units, n_pert, replace=False):
        side = rng.random(d[a].size) < 0.5
        d[a], d[new] = d[a][side], d[a][~side]
        new += 1
    return d


def p_dup():
    d, new = dict(d0), max(d0) + 1
    for a in rng.choice(units, n_pert, replace=False):
        d[new] = np.sort(jitter_copy(d0[a], 0.8))
        new += 1
    return d


lvl = {'baseline': L0}
for name, fn in (('merge_diff', p_merge), ('drop', p_drop), ('contaminate', p_contam), ('thin', p_thin),
                 ('split', p_split), ('duplicate_units', p_dup)):
    draws = [sort_level(fn()) for _ in range(args.draws)]
    lvl[name] = {k: float(np.mean([x[k] - L0[k] for x in draws])) for k in L0}
    lvl[name]['draws_dS'] = [x['S'] - L0['S'] for x in draws]
    print(name, {k: round(v, 4) for k, v in lvl[name].items() if k != 'draws_dS'}, f'({time.time() - t0:.0f} s)', flush=True)
res['sort_level'] = lvl
res['runtime_s'] = time.time() - t0
(common.OUT / f'perturb_{args.sort}.json').write_text(json.dumps(res, indent=1))
