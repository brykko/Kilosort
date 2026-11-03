"""Tables of the KS2.5 "standard_low" scoring (search_ks25.md section 8), with "standard" alongside: reads the
outputs of score_low.sh and the earlier KS2.5 comparison."""
import json
from pathlib import Path

import numpy as np

K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11/runs')
RUNS = {'standard': (231, 232, 233), 'standard_low': (234, 235, 236)}
TAG = {'standard': 'ks25', 'standard_low': 'ks25low'}
rd = lambda p: json.loads(Path(p).read_text())  # noqa: E731
ms = lambda a: f'{np.mean(a):7.1f} +- {np.std(a, ddof=1):5.1f}'  # noqa: E731

print('Kobe: KS2.5 against default (run_candidate --phase comparison)')
for preset, runs in RUNS.items():
    for k, r in enumerate(runs, 1):
        c = rd(K / f'search/{r}_ks25_{preset}_s{k}/summary.json')
        g = {x['guard']: x for x in c['guards']}
        fails = [x['guard'] for x in c['guards'] if x['status'] != 'pass']
        print(f'  {preset:12s} r{k}: S {c["S"]:7.1f}  dS {c["dS"]:+6.1f}  dS_track {c["dS_track"]:+6.1f}  eligible {c["n_eligible"]:4d}  '
              f'units {c["n_units"]:4d}  agreement {c["hybrid"]["median_agreement"]:.3f}  recovered {c["hybrid"]["n_recovered"]}  '
              f'untuned {g["Untuned units"]["candidate"]:.0f}  guards {"pass" if not fails else "FAIL: " + ", ".join(fails)}')
    S = [rd(K / f'search/{r}_ks25_{preset}_s{k}/summary.json')['S'] for k, r in enumerate(runs, 1)]
    St = [rd(K / f'search/{r}_ks25_{preset}_s{k}/summary.json')['S_track'] for k, r in enumerate(runs, 1)]
    print(f'  {preset:12s} mean: S {ms(S)}, S_track {ms(St)}')

print('\nKobe: H8 against KS2.5 (compare.py, KS2.5 replicates as baseline)')
for preset, tag in TAG.items():
    for k in (1, 2, 3):
        c = rd(K / f'ks25_comparison/H8_vs_{tag}_s{k}.json')
        fails = [x['guard'] for x in c['guards'] if x['status'] != 'pass']
        print(f'  vs {preset:12s} H8 s{k}: S {c["S"]:7.1f}  baseline {c["S_default"]:7.1f}  dS {c["dS"]:+6.1f}  dS_track {c["dS_track"]:+6.1f}  '
              f'matched {c["matched_units"]["n_matched"]}  guards {"pass" if not fails else "FAIL: " + ", ".join(fails)}')

print('\nKobe: units reproduced across replicates; stricter cleaning')
for preset, tag in TAG.items():
    rep = rd(K / f'reproducibility_{tag}.json')
    f = [x[k] for x in rep for k in ('matched_a_in_b', 'matched_b_in_a')]
    st = rd(K / f'strict_cleaning_{tag}.json')['sorts']
    print(f'  {preset:12s} reproduced {np.mean(f):.1%} (range {min(f):.1%}-{max(f):.1%}); strict S ' + ', '.join(f'{v["S_strict"]:.1f}' for v in st.values()))

print('\nMEC: tracked-position S')
for preset, tag in TAG.items():
    t = rd(M / f'transfer_{tag}.json')
    print(f'  {preset:12s} alpha {t["alpha_s"]}; default S {t["summary"]["S_default_mean"]:.1f} +- {t["summary"]["S_default_sd"]:.1f}')
    for w in t['winner']:
        if Path(w['sort']).name.startswith('ks25_'):      # transfer_ks25.json also scored H8 against default
            print(f'    {Path(w["sort"]).name:22s} S {w["S"]:7.1f}  dS {w["dS"]:+6.1f}  eligible {w["n_eligible"]}  units {w["n_units"]}  '
                  f'untuned {w["guards"]["untuned_units"][0]:.0f} (default {w["guards"]["untuned_units"][1]:.0f})  '
                  f'guards {"pass" if all(v[2] for v in w["guards"].values()) else "FAIL"}')
    h = rd(M / f'transfer_H8_vs_{tag}.json')
    print(f'    H8 against {preset}: baseline S {h["summary"]["S_default_mean"]:.1f} +- {h["summary"]["S_default_sd"]:.1f}; ' + ', '.join(
        f'{Path(w["sort"]).name} dS {w["dS"]:+.1f} ({"pass" if all(v[2] for v in w["guards"].values()) else "FAIL"})' for w in h['winner']))
    rep = rd(M / f'reproducibility_{tag}.json')
    f = [x[k] for x in rep for k in ('matched_a_in_b', 'matched_b_in_a')]
    print(f'    reproduced {np.mean(f):.1%}')

print('\nDecoding, median error (deg / cm), mean over replicates; v0.20 files (all four sorters on the same windows)')
for ex, root in (('kobe', K), ('MEC', M)):
    b, c = rd(root / 'bayes_decode_v020.json'), rd(root / 'corr_decode_v020.json')
    for what in ('hd', 'pos'):
        for win in ('0.02', '0.1'):
            row = []
            for dec, d, sub in (('bayes', b, None), ('raw', c, 'raw'), ('z', c, 'z')):
                vals = {g: np.mean([(x['median_error'][sub] if sub else x['median_error'])[what][win] for x in xs])
                        for g, xs in d['groups'].items()}
                row.append(f'{dec}: ' + ' '.join(f'{g} {v:5.1f}' for g, v in vals.items()))
            print(f'  {ex:4s} {what:3s} {win:4s}s  ' + ' | '.join(row))
    n = {g: np.mean([x['n_units'] for x in xs]) for g, xs in b['groups'].items()}
    print(f'  {ex} units decoded: ' + ', '.join(f'{g} {v:.0f}' for g, v in n.items()))
