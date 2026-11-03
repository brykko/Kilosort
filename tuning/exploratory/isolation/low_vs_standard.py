"""Are KS2.5 standard's units found again by standard_low, and are they as clean there? Seed 1 of each; as a
control for seed-to-seed change, the same comparison between seeds 1 and 2 of standard.

Units: all units of the cleaned sort, hybrid copies out (as tradeoff.py); eligibility as scored. Units are
paired one to one (agreement >= 0.5 within 100 um, greedy by agreement, as matched_contam.py). Per pair:
spikes, the share of standard's spikes that standard_low keeps in the partner (+-0.4 ms) and contamination
(plain and local, as comod.py). Pairs are grouped by the standard unit's plain contamination. "Extra
misassigned" is the estimated misassigned spikes the partner adds per spike it adds:
(sum n_low c_low - sum n_std c_std) / (sum n_low - sum n_std).

For standard's eligible units without a partner: the share of their spikes found in any standard_low unit,
and whether they went to a partner of another standard unit (merged) or over several units (split).
Then where standard_low's extra spikes are: in paired units, in units of its own, less standard's own units.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hybrid  # noqa: E402
from cleaning import load_clean_units  # noqa: E402
from comod import unit_estimates  # noqa: E402
from isolation import violations  # noqa: E402
from rate_mix import TAU  # noqa: E402
from compare import TOL_S  # noqa: E402
from matched_contam import pairs  # noqa: E402

K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
SORTS = {'kobe': (K / 'runs/ks25_standard_r1', K / 'runs/ks25_standard_low_r1'),
         'mec': (M / 'runs/ks25_standard_r1', M / 'runs/ks25_standard_low_r1')}
MEC_ELIG = {Path(e['sort']).as_posix(): [int(c) for c in e['S_i']]
            for f in ('transfer_ks25.json', 'transfer_ks25low.json')
            for e in json.loads((M / 'runs' / f).read_text())['winner']}
CLASSES = ((0, 0.02, '< 2%'), (0.02, 0.10, '2-10%'), (0.10, 2.0, '>= 10%'))


def load(sort, ex):
    base = K if ex == 'kobe' else M
    by, depth, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, np.load(base / 'search/hybrid/hybrid_truth.npz'))['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    T = 1200 - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else 1200.0
    elig = (json.loads((sort / 'evaluation.json').read_text())['S_i'] if ex == 'kobe' else MEC_ELIG[sort.as_posix()])
    units = {c: np.sort(t) for c, t in by.items() if t.size}
    est = {c: unit_estimates(t, T) for c, t in units.items()}
    return units, depth, est, {int(c) for c in elig}


def found(ta, tb):
    """Which spikes of ta have a spike of tb within TOL_S."""
    if tb.size == 0:
        return np.zeros(ta.size, bool)
    tb = np.concatenate(([-np.inf], tb, [np.inf]))
    i = np.searchsorted(tb, ta)
    return np.minimum(np.abs(tb[i] - ta), np.abs(tb[i - 1] - ta)) <= TOL_S


def misassigned(est, ids, m):
    return sum(est[c][0] * est[c][2][m] for c in ids)


def compare(sa, sb, ex, la, lb):
    us, ds, es, el_s = load(sa, ex)
    ul, dl, el, el_l = load(sb, ex)
    pr = pairs(us, ds, ul, dl)
    partner = dict(pr)
    print(f'\n===== {ex}: {la} {len(us)} units ({len(el_s)} eligible), {lb} {len(ul)} units '
          f'({len(el_l)} eligible); {len(pr)} one-to-one pairs')

    # a's eligible units: found again?
    e_pairs = [(a, b) for a, b in pr if a in el_s]
    print(f'{la} eligible units paired: {len(e_pairs)}/{len(el_s)} ({len(e_pairs) / len(el_s):.0%}); '
          f'partner eligible in {lb}: {sum(b in el_l for _, b in e_pairs)}')

    # pairs by a's contamination class (a's eligible units)
    print(f'\nPairs ({la} eligible), by {la} plain contamination:')
    print(f'{"class":>8s} {"pairs":>6s} {"agree":>6s} {"n_b/n_a":>11s} {"kept":>6s}   '
          f'{"plain a -> b":>18s} {"extra mis.":>10s} {"b dirtier >2pp":>16s}   '
          f'{"local a -> b":>18s} {"extra mis.":>10s}')
    for lo, hi, name in CLASSES + ((0, 2.0, 'all'),):
        P = [(a, b) for a, b in e_pairs if lo <= es[a][2]['plain'] < hi]
        if not P:
            continue
        A, B = [a for a, _ in P], [b for _, b in P]
        ns, nl = sum(es[a][0] for a in A), sum(el[b][0] for b in B)
        kept = np.median([found(us[a], ul[b]).mean() for a, b in P])
        ratio = np.median([el[b][0] / es[a][0] for a, b in P])
        agree = np.median([found(us[a], ul[b]).sum() / (us[a].size + ul[b].size - found(us[a], ul[b]).sum())
                           for a, b in P])
        row = f'{name:>8s} {len(P):6d} {agree:6.2f} {ratio:11.2f} {kept:6.1%}   '
        for m in ('plain', 'local'):
            cs, cl = misassigned(es, A, m) / ns, misassigned(el, B, m) / nl
            extra = (misassigned(el, B, m) - misassigned(es, A, m)) / (nl - ns) if nl != ns else np.nan
            row += f'{cs:8.1%} -> {cl:6.1%} {extra:10.1%} '
            if m == 'plain':
                worse = np.mean([el[b][2][m] - es[a][2][m] > 0.02 for a, b in P])
                row += f'{worse:16.0%}   '
        print(row)

    # per-pair spread of the change, clean units of a
    P = [(a, b) for a, b in e_pairs if es[a][2]['plain'] < 0.02]
    d = np.array([el[b][2]['plain'] - es[a][2]['plain'] for a, b in P])
    print(f'\nClean {la} units (< 2% plain, {len(P)} pairs): change in plain contamination, percentiles '
          f'10/50/90/99: ' + ', '.join(f'{x:+.1%}' for x in np.percentile(d, (10, 50, 90, 99)))
          + f'; partners at >= 5%: {np.sum([el[b][2]["plain"] >= 0.05 for _, b in P])}, at >= 10%: '
          f'{np.sum([el[b][2]["plain"] >= 0.10 for _, b in P])}')
    v = np.array([(violations(us[a], TAU), violations(ul[b], TAU)) for a, b in P])
    hi = np.array([el[b][2]['plain'] >= 0.10 for _, b in P])
    print(f'  refractory violations (< {TAU * 1e3:g} ms) in the partner: {np.mean(v[:, 1] == 0):.0%} none, '
          f'{np.mean((v[:, 1] >= 1) & (v[:, 1] <= 2)):.0%} one or two, {np.mean(v[:, 1] >= 3):.0%} three or more '
          f'({la} units: {np.mean(v[:, 0] == 0):.0%} none); partners at >= 10% plain: median spikes '
          f'{np.median([el[b][0] for (_, b), h in zip(P, hi) if h]):.0f}, median violations {np.median(v[hi, 1]):.0f} '
          f'(max {v[hi, 1].max()}), median local {np.median([el[b][2]["local"] for (_, b), h in zip(P, hi) if h]):.1%}')

    # a's eligible units without a partner
    lone = [a for a in el_s if a not in partner]
    taken = {b: a for a, b in pr}
    lt = sorted(ul)
    kind = {'merged': [], 'grown': [], 'split': [], 'lost': []}
    for a in lone:
        hits = {b: found(us[a], ul[b]).mean() for b in lt if abs(dl[b] - ds[a]) <= 100}
        best = max(hits, key=hits.get) if hits else None
        if best is not None and hits[best] >= 0.5:
            kind['merged' if best in taken else 'grown'].append((a, best))
        elif sum(hits.values()) >= 0.5:
            kind['split'].append((a, best))
        else:
            kind['lost'].append((a, best))
    print(f'\n{la} eligible units without a partner: {len(lone)} (median plain contamination '
          f'{np.median([es[a][2]["plain"] for a in lone]) if lone else np.nan:.1%}). Where their spikes are:')
    print(f'  merged: half or more in one {lb} unit paired with another {la} unit: {len(kind["merged"])}')
    g = kind['grown']
    print(f'  grown: half or more in one {lb} unit with no partner: {len(g)}'
          + (f' (median spikes {la} -> {lb} unit: x{np.median([el[b][0] / es[a][0] for a, b in g]):.2f}; '
             f'plain {misassigned(es, [a for a, _ in g], "plain") / sum(es[a][0] for a, _ in g):.1%} -> '
             f'{misassigned(el, [b for _, b in g], "plain") / sum(el[b][0] for _, b in g):.1%}, '
             f'local {misassigned(es, [a for a, _ in g], "local") / sum(es[a][0] for a, _ in g):.1%} -> '
             f'{misassigned(el, [b for _, b in g], "local") / sum(el[b][0] for _, b in g):.1%})' if g else ''))
    print(f'  split: spread over several {lb} units: {len(kind["split"])}; lost: under half in any {lb} unit '
          f'nearby: {len(kind["lost"])}')

    # where b's extra spikes are (eligible units)
    Pa = [(a, b) for a, b in pr if a in el_s or b in el_l]
    pa_s, pa_l = [a for a, _ in Pa if a in el_s], [b for _, b in Pa if b in el_l]
    own_l = [b for b in el_l if b not in {b for _, b in Pa}]
    own_s = [a for a in el_s if a not in {a for a, _ in Pa}]
    print(f'\nEligible spikes (millions), {la} -> {lb}, and estimated misassigned (plain / local):')
    for name, S, L in (('paired units', pa_s, pa_l), ('own units', own_s, own_l)):
        ns, nl = sum(es[c][0] for c in S) / 1e6, sum(el[c][0] for c in L) / 1e6
        print(f'  {name:13s} {len(S):4d} -> {len(L):4d} units, {ns:5.2f} -> {nl:5.2f} M spikes; misassigned '
              + '; '.join(f'{m} {misassigned(es, S, m) / 1e6:.3f} -> {misassigned(el, L, m) / 1e6:.3f} M'
                          for m in ('plain', 'local')))


if __name__ == '__main__':
    for ex, (s_std, s_low) in SORTS.items():
        compare(s_std, s_low, ex, 'standard', 'standard_low')
        compare(s_std, s_std.with_name('ks25_standard_r2'), ex, 'standard s1', 'standard s2')
