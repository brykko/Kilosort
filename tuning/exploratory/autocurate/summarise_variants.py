"""Tables of score_variants.sh: S of each cut sort and of its random controls, against the uncut sort (seed 1).

Per sort: spikes after cleaning (all units; kobe without hybrid copies), S (kobe also S_track), eligible
units, guards. Relative to the uncut sort: the share of spikes kept and of S kept. Then within units, for
units eligible both uncut and after the cut (unit ids are kept by the cut): their share of spikes kept and of
S kept; the random control removes the same number of spikes from each unit, so a cut that removes less
informative spikes than random keeps more of S there. S lost with units that stop being eligible is
reported separately.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hybrid  # noqa: E402
from cleaning import load_clean_units  # noqa: E402

K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
A = Path('C:/data/ks4_tuning/exploratory/autocurate')
V = A / 'variants'
CUTS = {'H8': ('cut2', 'cut4'), 'default': ('cut4',)}
TRUTH = np.load(K / 'search/hybrid/hybrid_truth.npz')


def spikes(sort, ex):
    by, _, _ = load_clean_units(sort, (K if ex == 'kobe' else M) / 'search/raw_artefacts_car.npz')
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, TRUTH)['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    return {c: t.size for c, t in by.items()}


def kobe_scores(sort):
    ev = json.loads((sort / 'evaluation.json').read_text())
    cmp = json.loads((sort / 'comparison.json').read_text())
    fails = [g['guard'] for g in cmp['guards'] if g['status'] not in ('pass', 'no tau')]
    return {'S': ev['S'], 'S_track': ev['S_track'], 'n_eligible': ev['n_eligible'],
            'S_i': {int(k): v for k, v in ev['S_i'].items()}, 'guards': 'pass' if not fails else 'FAIL: ' + ', '.join(fails)}


def mec_scores():
    t = json.loads((V / 'transfer_mec.json').read_text())
    out = {}
    for w in t['winner']:
        ok = all(v[2] for v in w['guards'].values())
        out[Path(w['sort']).name] = {'S': float(w['S']), 'n_eligible': int(w['n_eligible']),
                                     'S_i': {int(k): v for k, v in w['S_i'].items()},
                                     'guards': 'pass' if ok else 'FAIL: ' + ', '.join(k for k, v in w['guards'].items() if not v[2])}
    return out


def within(base, var, n0, n1):
    """Units eligible in both: share of spikes kept, share of S kept; and S lost with units no longer eligible."""
    both = [c for c in base['S_i'] if c in var['S_i']]
    sp = sum(n1.get(c, 0) for c in both) / sum(n0[c] for c in both)
    s = sum(var['S_i'][c] for c in both) / sum(base['S_i'][c] for c in both)
    lost = sum(v for c, v in base['S_i'].items() if c not in var['S_i'])
    return sp, s, lost


if __name__ == '__main__':
    mec = mec_scores()
    for ex in ('kobe', 'mec'):
        print(f'\n===== {ex}' + (' (S and S_track, bits/s)' if ex == 'kobe' else ' (tracked-position S, bits/s)'))
        for g, cuts in CUTS.items():
            base_dir = A / f'sorts/{ex}_{g}_s1'
            base = kobe_scores(base_dir) if ex == 'kobe' else mec[base_dir.name]
            n0 = spikes(base_dir, ex)
            N0 = sum(n0.values())
            print(f'\n  {g} uncut: S {base["S"]:.1f}' + (f', S_track {base["S_track"]:.1f}' if ex == 'kobe' else '')
                  + f', {base["n_eligible"]} eligible units, {N0 / 1e6:.2f} M spikes, guards {base["guards"]}')
            print(f'  {"":16s} {"spikes kept":>11s} {"S":>8s} {"S kept":>7s} ' + (f'{"S_track kept":>12s} ' if ex == 'kobe' else '')
                  + f'{"eligible":>8s}   {"within units: spikes / S kept":>30s} {"S lost, units dropped":>22s}  guards')
            for cut in cuts:
                rows = {}
                for name in [cut] + [f'{cut}_rand{s}' for s in (1, 2, 3)]:
                    d = V / f'{ex}_{g}_{name}'
                    sc = kobe_scores(d) if ex == 'kobe' else mec[d.name]
                    n1 = spikes(d, ex)
                    rows[name] = (sum(n1.values()) / N0, sc, within(base, sc, n0, n1))
                for name, (fs, sc, (wsp, ws, lost)) in rows.items():
                    print(f'  {name:16s} {fs:11.1%} {sc["S"]:8.1f} {sc["S"] / base["S"]:7.1%} '
                          + (f'{sc["S_track"] / base["S_track"]:12.1%} ' if ex == 'kobe' else '')
                          + f'{sc["n_eligible"]:8d}   {wsp:14.1%} / {ws:6.1%}         {lost:12.1f}  {sc["guards"]}')
                r = [rows[f'{cut}_rand{s}'] for s in (1, 2, 3)]
                S_r = np.array([x[1]['S'] for x in r])
                ws_r = np.array([x[2][1] for x in r])
                print(f'  {cut} against random: S {rows[cut][1]["S"]:.1f} vs {S_r.mean():.1f} +- {S_r.std(ddof=1):.1f} '
                      f'({(rows[cut][1]["S"] - S_r.mean()) / S_r.mean():+.1%}); within units S kept {rows[cut][2][1]:.1%} vs '
                      f'{ws_r.mean():.1%} +- {ws_r.std(ddof=1):.1%}', flush=True)
