"""Exploratory prototype (not part of the frozen harness): spikes in well-isolated units, without a cut-off.

Per unit of the cleaned sort (§6.1-6.3; hybrid copies excluded on the hybrid excerpt): refractory
violations V = pairs of the unit's own spikes with tau_c < interval < tau_r; contamination c solved from
V = 2 (tau_r - tau_c) N^2 c (1 - c/2) / T (Hill et al. 2011; censored as Llobet et al. 2022), T the
unmasked duration. Score: sum over units of N (1 - c), the estimated number of correctly assigned spikes;
also over eligible units only. tau_c = 0.25 ms (KS4's duplicate_spike_ms); tau_r = 1.5 and 2 ms.

Hybrid check (kobe): for each inserted unit's best-matching sorted unit, the estimated contamination
against the true one (fraction of its spikes not within 0.4 ms of an inserted spike).
Audit (default seed 1): splits, merges, drops and added contamination of 10% of units.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hybrid  # noqa: E402
import score  # noqa: E402
import waveform  # noqa: E402
from cleaning import load_clean_units, load_sort  # noqa: E402

TAU_C = 0.25e-3
TAU_R = (1.5e-3, 2.0e-3)
DUR = 1200.0
DATA = {'kobe': (Path('C:/data/ks4_tuning/26648_kobe_2020-08-21'), 'search/hybrid/hybrid_truth.npz'),
        'mec': (Path('C:/data/ks4_tuning/25843_roger_2019-09-11'), None)}


def violations(t, tau_r):
    t = np.sort(t)
    return int(np.sum(np.searchsorted(t, t + tau_r, 'left') - np.searchsorted(t, t + TAU_C, 'right')))


def contamination(n, v, T, tau_r):
    a = v * T / (2 * (tau_r - TAU_C) * max(n, 1) ** 2)
    return 1 - np.sqrt(1 - 2 * a) if a <= 0.5 else 1.0


def yield_of(units, T, eligible=None):
    out = {}
    for tau in TAU_R:
        c = np.array([contamination(t.size, violations(t, tau), T, tau) for t in units.values()])
        n = np.array([t.size for t in units.values()])
        key = f'{tau * 1e3:.1f}ms'
        out[key] = {'spikes': int(n.sum()), 'misassigned': float((n * c).sum()), 'score': float((n * (1 - c)).sum())}
        if eligible is not None:
            e = np.array([k in eligible for k in units])
            out[key]['score_eligible'] = float((n * (1 - c))[e].sum())
            out[key]['spikes_eligible'] = int(n[e].sum())
    return out


def load(sort, ex):
    base, truth = DATA[ex]
    raw = base / 'search/raw_artefacts_car.npz'
    by, _, iv, amp = load_clean_units(sort, raw, np.load(Path(sort) / 'amplitudes.npy').ravel())
    copies = set()
    if truth:
        copies = set(hybrid.recovery(sort, np.load(base / truth))['hybrid_copies'])
    by = {c: t for c, t in by.items() if c not in copies and t.size}
    T = DUR - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else DUR
    elig = set(score.eligible(sort, by, DUR, waveform.screen(sort)))
    _, _, ids, xy = load_sort(sort)
    pos = dict(zip(ids.tolist(), xy))
    return by, {c: amp[c] for c in by}, {c: pos[c] for c in by}, elig, T


def hybrid_check(sort, ex):
    base, truth = DATA[ex]
    tr = np.load(base / truth)
    raw = base / 'search/raw_artefacts_car.npz'
    by, _, _ = load_clean_units(sort, raw)
    rec = hybrid.recovery(sort, tr)
    tt = tr['spike_samples'] / 30000
    rows = []
    for u in rec['per_unit']:
        c = u['best_sorted']
        t = np.sort(by.get(c, np.array([])))
        if t.size < 50:
            continue
        truth_t = np.sort(tt[tr['spike_unit'] == u['inserted']])
        i = np.clip(np.searchsorted(truth_t, t), 1, truth_t.size - 1)
        hit = np.minimum(np.abs(truth_t[i] - t), np.abs(truth_t[i - 1] - t)) <= 0.4e-3
        true_c = 1 - hit.mean()
        est = {f'{tau * 1e3:.1f}ms': contamination(t.size, violations(t, tau), DUR, tau) for tau in TAU_R}
        rows.append({'inserted': u['inserted'], 'n': int(t.size), 'rate': t.size / DUR, 'true': float(true_c), **est})
    return rows


def perturb(by, amp, pos, mode, rng, frac=0.1):
    keys = list(by)
    pick = set(rng.permutation(len(keys))[:int(frac * len(keys))].tolist())
    out, new = {}, max(keys) + 1
    if mode == 'merge':
        xy = np.array([pos[k] for k in keys])
        used = set()
        for a in rng.permutation(len(keys)):
            if len(used) >= frac * len(keys) or a in used:
                continue
            d = np.hypot(*(xy - xy[a]).T)
            nb = [b for b in np.flatnonzero((d <= 100) & (np.arange(len(keys)) != a)) if b not in used]
            if nb:
                b = rng.choice(nb)
                used |= {a, b}
                out[new] = np.sort(np.concatenate([by[keys[a]], by[keys[b]]])); new += 1
        out.update({keys[k]: by[keys[k]] for k in range(len(keys)) if k not in used})
        return out
    for k, c in enumerate(keys):
        t = by[c]
        if k not in pick:
            out[c] = t
        elif mode == 'drop':
            continue
        elif mode == 'contaminate':
            other = by[keys[rng.integers(len(keys))]]
            extra = rng.choice(other, max(1, int(0.1 * t.size)), replace=other.size < 0.1 * t.size)
            out[c] = np.sort(np.concatenate([t, extra]))
        else:
            part = rng.integers(2, size=t.size) if mode == 'random' else (amp[c] > np.median(amp[c])).astype(int)
            out[c], out[new] = t[part == 0], t[part == 1]
            new += 1
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--excerpt', choices=list(DATA), required=True)
    p.add_argument('--sort', nargs=2, action='append', metavar=('LABEL', 'FOLDER'), required=True)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--hybrid', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    res = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for label, folder in args.sort:
        by, amp, pos, elig, T = load(folder, args.excerpt)
        r = {'n_units': len(by), 'n_eligible': len(elig), 'T': T, 'yield': yield_of(by, T, elig)}
        y = r['yield']['1.5ms']
        print(f"{label:12s} units {len(by):5d}  spikes {y['spikes']:9d}  misassigned {y['misassigned']:9.0f} "
              f"({y['misassigned'] / y['spikes']:.2%})  score {y['score']:9.0f}  eligible-only {y['score_eligible']:9.0f}", flush=True)
        if args.hybrid:
            r['hybrid'] = hybrid_check(folder, args.excerpt)
        if args.audit:
            r['audit'] = {}
            for mode in ('random', 'amplitude', 'merge', 'drop', 'contaminate'):
                pb = perturb(by, amp, pos, mode, rng)
                r['audit'][mode] = yield_of(pb, T)
                d = r['audit'][mode]['1.5ms']
                print(f"   {mode:12s} score {d['score'] - y['score']:+9.0f}  misassigned {d['misassigned'] - y['misassigned']:+8.0f}  "
                      f"spikes {d['spikes'] - y['spikes']:+8d}", flush=True)
        res[label] = r
        Path(args.out).write_text(json.dumps(res, indent=1))
