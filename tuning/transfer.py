"""Transfer test (spec §10): score default and winner sorts of the MEC excerpt with tracked-position S.

Each sort is cleaned (§6.1-6.3), its eligible units (§6.4) are scored with position_score, and the
winner is compared with default on S and on two guards (§7.1): eligible units and untuned units.

alpha (Appendix A) is set on the default sorts before the winner is scored: the value among
ALPHAS with the highest mean cross-validated S without the null correction, i.e. the regulariser that
best predicts held-out spiking. At that alpha, the time-shift control scores the default sorts with
the tracking circularly shifted by SHIFT_S; S should be within 2 SD of default's S across seeds of 0.

    python transfer.py --default <sort> ... --winner <sort> ... --tracking <npz> --raw-artefacts <npz>
        --t0 13575 --duration 1200 --out <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np

import position_score
import waveform
from cleaning import load_clean_units, run_cleaning
from score import eligible

ALPHAS = [0.0625, 0.25, 1.0, 4.0, 16.0]
SHIFT_S = 300.0


def prepare(sort, raw_artefacts, duration):
    sort = Path(sort)
    if not (sort / 'cleaning.npz').exists():
        run_cleaning(sort, duration)
    by_cluster, _, mask = load_clean_units(sort, raw_artefacts)
    screen = waveform.screen(sort)
    return {'sort': str(sort), 'by': by_cluster, 'mask': mask, 'units': eligible(sort, by_cluster, duration, screen),
            'n_units': len(by_cluster), 'n_noise_units': sum(not v[1] for v in screen.values()),
            'multiunit_events': json.loads((sort / 'cleaning.json').read_text())['multiunit_events']['n_events']}


def shifted(tracking, seconds):
    """Tracking with every field but time circularly shifted by `seconds`."""
    k = int(round(seconds / np.median(np.diff(tracking['t']))))
    return {f: (tracking[f] if f == 't' else np.roll(tracking[f], k)) for f in tracking}


def untuned(s_i, q):
    return int(np.sum(np.array(list(s_i.values())) < q))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--default', nargs='+', required=True)
    p.add_argument('--winner', nargs='+', required=True)
    p.add_argument('--tracking', required=True)
    p.add_argument('--raw-artefacts', required=True)
    p.add_argument('--t0', type=float, required=True)
    p.add_argument('--duration', type=float, required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    tracking = dict(np.load(args.tracking))
    run = lambda s, a, null=True, trk=tracking: position_score.score_sort(
        s['by'], s['units'], trk, args.t0, args.duration, s['mask'], a, null)

    defaults = [prepare(s, args.raw_artefacts, args.duration) for s in args.default]
    scan = {a: [run(s, a, null=False)['S'] for s in defaults] for a in ALPHAS}
    for a, v in scan.items():
        print(f'alpha {a:5.2f} s: uncorrected S {np.mean(v):7.1f} ({", ".join(f"{x:.1f}" for x in v)})', flush=True)
    alpha = max(ALPHAS, key=lambda a: np.mean(scan[a]))
    print(f'alpha = {alpha} s', flush=True)

    res_def = [run(s, alpha) for s in defaults]
    control = [run(s, alpha, trk=shifted(tracking, SHIFT_S))['S'] for s in defaults]
    S_def = np.array([r['S'] for r in res_def])
    q = [np.percentile(list(r['S_i'].values()), 25) for r in res_def]
    unt_def = np.mean([untuned(r['S_i'], qq) for r, qq in zip(res_def, q)])
    n_def = np.mean([len(s['units']) for s in defaults])
    print(f'default S {S_def.mean():.1f} +- {S_def.std(ddof=1):.1f} ({", ".join(f"{x:.1f}" for x in S_def)}); '
          f'eligible {n_def:.0f}; time-shift control {", ".join(f"{x:+.1f}" for x in control)}', flush=True)

    out = {'alpha_s': alpha, 'alpha_scan_uncorrected_S': {str(a): v for a, v in scan.items()},
           'time_shift_s': SHIFT_S, 'time_shift_control_S': control, 'default': [], 'winner': []}
    for s, r in zip(defaults, res_def):
        out['default'].append({'sort': s['sort'], 'S': r['S'], 'n_eligible': len(s['units']), 'n_units': s['n_units'],
                               'n_noise_units': s['n_noise_units'], 'multiunit_events': s['multiunit_events'],
                               'S_i': {str(c): v for c, v in r['S_i'].items()}})
    for w in args.winner:
        s = prepare(w, args.raw_artefacts, args.duration)
        r = run(s, alpha)
        unt = float(np.mean([untuned(r['S_i'], qq) for qq in q]))
        rec = {'sort': s['sort'], 'S': r['S'], 'dS': r['S'] - S_def.mean(), 'n_eligible': len(s['units']),
               'n_units': s['n_units'], 'n_noise_units': s['n_noise_units'], 'multiunit_events': s['multiunit_events'],
               'guards': {'eligible_units': [len(s['units']), float(n_def), bool(len(s['units']) >= 0.8 * n_def)],
                          'untuned_units': [unt, float(unt_def), bool(unt >= 0.9 * unt_def)]},
               'S_i': {str(c): v for c, v in r['S_i'].items()}}
        out['winner'].append(rec)
        print(f"{w}: S {r['S']:.1f} (dS {rec['dS']:+.1f}), eligible {len(s['units'])} (default {n_def:.0f}), "
              f"untuned {unt:.0f} (default {unt_def:.0f}); guards "
              f"{'pass' if all(g[2] for g in rec['guards'].values()) else 'FAIL'}", flush=True)
    dS = [w['dS'] for w in out['winner']]
    out['summary'] = {'S_default_mean': float(S_def.mean()), 'S_default_sd': float(S_def.std(ddof=1)),
                      'dS_mean': float(np.mean(dS)), 'dS_over_default_sd': float(np.mean(dS) / S_def.std(ddof=1))}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out['summary']))
