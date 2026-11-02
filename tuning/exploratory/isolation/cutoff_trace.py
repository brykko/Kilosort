"""Why KS2.5 doesn't output the extra spikes in KS4's units: never extracted, or deleted by set_cutoff?

Uses the traced KS2.5 reruns (ks25_setup/ks25_sort_traced.py, seed 1), compared only with their own
stages (KS2.5 is not deterministic). Spike sets of the rerun:
- final: the cleaned sort (as every other sort here);
- cut: spikes deleted by set_cutoff (cycle 1, strict, and cycle 2), grouped by template;
- extracted: every spike learnAndSolve8b extracted (score >= Th(2) = 6), grouped by template.
Template positions follow KS4's rule on the final rezToPhy templates (ids are stable through autocurate).

1. KS2.5's own losses: spikes extracted, deleted by each cut, removed by our cleaning; contamination of
   its units before the cuts (stage split1), between them (split2) and after (merge3), all computed the
   same way (masked intervals dropped, no duplicate cleaning), so the three are comparable.
2. For KS4 (default, H8; seed 1) eligible units, each spike not in the rerun's final output (labels as
   trace_units.py) is classed: deleted by set_cutoff (coincides with a cut pseudo-unit), extracted but
   otherwise not output (e.g. removed by our cleaning), or never extracted. The same split for the
   spikes in refractory violations, i.e. KS4's misassigned spikes, shared out as in trace_units.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cleaning import load_clean_units, mask_intervals  # noqa: E402
from ks25_sort import PAD, unit_positions  # noqa: E402
from trace_units import SORTS, contamination, label, load, violation_pairs  # noqa: E402

FS = 30000
DUR = 1200.0
K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
RERUN = Path('C:/data/ks4_tuning/exploratory/ks25_traced')
CATS = ('deleted by set_cutoff', 'extracted, not output', 'never extracted')


def group(t, clu, keep=None):
    if keep is not None:
        t, clu = t[keep], clu[keep]
    o = np.argsort(clu, kind='stable')
    ids, start = np.unique(clu[o], return_index=True)
    return {int(c): np.sort(x) for c, x in zip(ids, np.split(t[o], start[1:]))}


def removed(st, a, b):
    """Rows of stage a missing from stage b (b is a with rows deleted, order kept)."""
    ka = st[f'{a}_t'].astype(np.int64) * 65536 + st[f'{a}_clu']
    kb = st[f'{b}_t'].astype(np.int64) * 65536 + st[f'{b}_clu']
    return ~np.isin(ka, kb)


def unmasked(t, iv):
    i = np.searchsorted(iv[:, 0], t, 'right') - 1
    return ~((i >= 0) & (t < iv[np.maximum(i, 0), 1])) if iv.size else np.ones(t.size, bool)


def stage_contamination(units, T):
    n = np.array([t.size for t in units.values()], float)
    c = [contamination(t, T) for t in units.values()]
    return {m: (np.sum(n * np.array([x[m] for x in c])) / n.sum(), n.sum()) for m in ('plain', 'local')}


for ex in ('kobe', 'mec'):
    base = K if ex == 'kobe' else M
    raw = base / 'search/raw_artefacts_car.npz'
    R = RERUN / f'{ex}_s1'
    st = np.load(R / 'stages.npz')
    final_times = np.load(R / 'spike_times.npy').ravel()
    assert np.array_equal(np.sort(st['merge3_t'] - 1).astype(np.int64), np.sort(final_times).astype(np.int64)), 'stage/export mismatch'
    W = np.load(R / 'templates_rezToPhy.npy')[:, PAD:]
    xy = np.load(R / 'channel_positions.npy')
    with np.errstate(invalid='ignore'):
        y = unit_positions(W, xy)[:, 1]
    tdepth = {k + 1: float(v) for k, v in enumerate(y) if np.isfinite(v)}   # st3 template ids are 1-based
    iv = mask_intervals(R, raw)
    T = DUR - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else DUR
    sec = {s: (st[f'{s}_t'] - 1) / FS for s in ('extracted', 'split1', 'cut1', 'split2', 'cut2', 'merge3')}

    # 1. KS2.5's own losses
    fin_by, fin_depth, _ = load_clean_units(R, raw)
    fin_by = {c: np.sort(t) for c, t in fin_by.items() if t.size}
    n_ext, n_c1, n_c2 = st['extracted_t'].size, removed(st, 'split1', 'cut1').sum(), removed(st, 'split2', 'cut2').sum()
    n_fin_raw, n_fin = st['merge3_t'].size, sum(t.size for t in fin_by.values())
    print(f'\n=== {ex}: KS2.5 rerun ===\n  extracted {n_ext / 1e6:.2f} M; set_cutoff deleted {n_c1 / 1e6:.2f} M (cycle 1) + '
          f'{n_c2 / 1e6:.2f} M (cycle 2) = {(n_c1 + n_c2) / n_ext:.0%}; exported {n_fin_raw / 1e6:.2f} M; after our cleaning {n_fin / 1e6:.2f} M')
    for s in ('split1', 'split2', 'merge3'):
        keep = unmasked(sec[s], iv)
        r = stage_contamination(group(sec[s], st[f'{s}_clu'], keep), T)
        print(f'  stage {s:7s}: {r["plain"][1] / 1e6:.2f} M spikes (masked out), misassigned {r["plain"][0]:.1%} plain / {r["local"][0]:.1%} local; '
              f'correctly assigned {r["plain"][1] * (1 - r["plain"][0]) / 1e6:.2f} / {r["local"][1] * (1 - r["local"][0]) / 1e6:.2f} M')
    m1, m2 = removed(st, 'split1', 'cut1'), removed(st, 'split2', 'cut2')
    cut_t = np.concatenate([sec['split1'][m1], sec['split2'][m2]])
    cut_c = np.concatenate([st['split1_clu'][m1], st['split2_clu'][m2]])
    cut_by = {c: x for c, x in group(cut_t, cut_c).items() if c in tdepth}
    ext_by = {c: x for c, x in group(sec['extracted'], st['extracted_clu']).items() if c in tdepth}
    d_cut = {c: tdepth[c] for c in cut_by}
    d_ext = {c: tdepth[c] for c in ext_by}

    # 2. KS4's spikes not in the rerun's output
    for g in ('default', 'H8'):
        by4, d4, T4, el4 = load(SORTS[ex][g], ex)
        e4 = {c: by4[c] for c in el4}
        lab = label(e4, d4, fin_by, fin_depth, T)
        miss = {c: e4[c][lab[c] < 0] for c in e4}
        lab_cut = label(miss, d4, cut_by, d_cut, T)
        rest = {c: miss[c][lab_cut[c] < 0] for c in miss}
        lab_ext = label(rest, d4, ext_by, d_ext, T)
        cat = {}
        for c in e4:
            k = np.full(e4[c].size, -1)
            idx = np.flatnonzero(lab[c] < 0)
            k[idx] = 2
            k[idx[lab_cut[c] >= 0]] = 0
            idx2 = idx[lab_cut[c] < 0]
            k[idx2[lab_ext[c] >= 0]] = 1
            cat[c] = k
        n_all = sum(t.size for t in e4.values())
        n_miss = np.bincount(np.concatenate([k[k >= 0] for k in cat.values()]), minlength=3)
        mis = {m: np.zeros(4) for m in ('plain', 'local')}       # the three classes, then 'in KS2.5' (both spikes output)
        for c, t in e4.items():
            i, j = violation_pairs(t)
            if not i.size:
                continue
            ki, kj = cat[c][i], cat[c][j]
            # a pair's class: its spike not in KS2.5's output (the earlier stage if both are missing)
            pc = np.where(ki < 0, kj, np.where(kj < 0, ki, np.minimum(ki, kj)))
            pc = np.where(pc < 0, 3, pc)
            frac = np.bincount(pc, minlength=4) / pc.size
            est = contamination(t, T4)
            for m in mis:
                mis[m] += t.size * est[m] * frac
        print(f'  {g}: {n_all / 1e6:.2f} M spikes in eligible units, {n_miss.sum() / 1e6:.2f} M ({n_miss.sum() / n_all:.0%}) not in the rerun\'s output: '
              + ', '.join(f'{cn} {x / 1e6:.2f} M ({x / n_miss.sum():.0%})' for cn, x in zip(CATS, n_miss)))
        for m, v in mis.items():
            print(f'     misassigned ({m}) {v.sum() / 1e6:.2f} M: ' + ', '.join(
                f'{cn} {x / 1e6:.2f} M ({x / v.sum():.0%})' for cn, x in zip(CATS + ('both spikes in KS2.5',), v)), flush=True)
