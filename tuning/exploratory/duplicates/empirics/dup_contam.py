"""How much of the KS4-vs-KS2.5 gap in refractory-violation contamination comes from within-unit duplicates?

For every sort (harness-cleaned units, hybrid copies out on kobe, all units: the 'plain' estimate of
tuning/exploratory/isolation/comod.py and rate_mix.py) and duplicate window w in 0.25/0.5/0.75/1.0 ms:
remove within-unit duplicates with KS4's rule (drop a spike within int(w*30) samples of the unit's last kept
spike, postprocessing.py:9-29), then solve Hill et al. (2011) with tau_c = w, tau_r = 1.5 ms. Reported: %
of spikes misassigned (sum N c / sum N), spikes removed by the dedup, per sorter and replicate.
Second part: the same on units matched one-to-one between each KS4 sorter (seed 1) and KS2.5 standard r1
(matched_contam.py's matcher: eligible units, agreement >= 0.5, within 100 um).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dup_common import FS, SORTS, TRUTH, ks4_dedup  # noqa: E402

REPO = Path('C:/code/github/Kilosort/tuning')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'exploratory/isolation'))
import hybrid  # noqa: E402
from cleaning import load_clean_units  # noqa: E402

TAU_R = 1.5e-3
WINDOWS = (0.25, 0.5, 0.75, 1.0)


def hill(n, v, T, tau_c, tau_r=TAU_R):
    a = v * T / (2 * (tau_r - tau_c) * max(n, 1) ** 2)
    return 1 - np.sqrt(1 - 2 * a) if a <= 0.5 else 1.0


def unit_contam(t_s, T):
    """t_s: sorted spike times (s). Returns {w: (n_after, c)}."""
    out = {}
    ts = np.round(t_s * FS).astype(np.int64)
    for w in WINDOWS:
        dt = int(w * FS / 1000)
        keep = ks4_dedup(ts, np.zeros(ts.size, np.int64), dt) if ts.size else np.zeros(0, bool)
        t = t_s[keep]
        v = int(np.sum(np.searchsorted(t, t + TAU_R, 'left') - np.searchsorted(t, t + w * 1e-3, 'right')))
        out[w] = (t.size, hill(t.size, v, T, w * 1e-3), ts.size)
    return out


def sort_units(sort, ex):
    base = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21' if ex == 'kobe' else 'C:/data/ks4_tuning/25843_roger_2019-09-11')
    by, _, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, np.load(TRUTH))['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    T = 1200 - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else 1200.0
    return {c: np.sort(t) for c, t in by.items() if t.size}, T


def summary(res):
    """res: list of per-unit dicts. Misassigned share at each w, relative to the original spike count."""
    n0 = np.array([r[0.25][2] for r in res], float)
    row = {}
    for w in WINDOWS:
        n = np.array([r[w][0] for r in res], float); c = np.array([r[w][1] for r in res])
        row[w] = ((n * c).sum() / n.sum(), 1 - n.sum() / n0.sum())
    return row


if __name__ == '__main__':
    print('Part 1: all units (cleaned, hybrid copies out). % misassigned (Hill, tau_c = w, tau_r = 1.5 ms) after '
          'KS4-rule dedup at w; [% spikes removed by that dedup]')
    print(f"{'':22s}" + ''.join(f'{f"w={w}ms":>18s}' for w in WINDOWS))
    for ex, groups in SORTS.items():
        for g, sorts in groups.items():
            rows = []
            for s in sorts:
                by, T = sort_units(s, ex)
                rows.append(summary([unit_contam(t, T) for t in by.values()]))
            for k, row in enumerate(rows, 1):
                print(f'{ex:5s}{g:12s} r{k}   ' + ''.join(f'{row[w][0]:9.1%} [{row[w][1]:5.2%}]' for w in WINDOWS), flush=True)
            print(f'{ex:5s}{g:12s} mean ' + ''.join(f'{np.mean([r[w][0] for r in rows]):9.1%} [{np.mean([r[w][1] for r in rows]):5.2%}]'
                                                  for w in WINDOWS), flush=True)

    print('\nPart 2: units matched one-to-one with KS2.5 standard r1 (seed/replicate 1 of each sorter)')
    import matched_contam as mc  # noqa: E402
    ks25 = {'kobe': SORTS['kobe']['ks25_std'][0], 'mec': SORTS['mec']['ks25_std'][0]}
    for ex in ('kobe', 'mec'):
        ub, db, Tb = mc.load(ks25[ex], ex)
        for g in ('ks4_default', 'ks4_H8') + (('ks25_low',) if ex == 'kobe' else ()):  # no MEC eligibility list for ks25_low
            ua, da, Ta = mc.load(SORTS[ex][g][0], ex)
            pr = mc.pairs(ua, da, ub, db)
            ra = [unit_contam(ua[a], Ta) for a, _ in pr]
            rb = [unit_contam(ub[b], Tb) for _, b in pr]
            sa, sb = summary(ra), summary(rb)
            print(f'{ex:5s}{g:12s} vs ks25_std, {len(pr)} cells: ' + '  '.join(
                f'w={w}: {sa[w][0]:5.1%} vs {sb[w][0]:5.1%} (gap {sa[w][0] - sb[w][0]:+5.1%})' for w in WINDOWS), flush=True)
