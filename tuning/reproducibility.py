"""Cross-seed reproducibility of units: what fraction of one seed's eligible units find a match among
another seed's eligible units of the same configuration. Real cells reappear in every seed; fragments
of over-split cells depend on the seed, so the fraction falls once resolution passes the real cells.
Uses spike trains, positions and eligibility only (no scores).

A match is spike-train agreement >= 0.5 (matched / (n_a + n_b - matched), +-0.4 ms) between units
whose median depths are <= 100 um apart, as in compare.py.

    python reproducibility.py --pair <sort seed A> <sort seed B> [--pair ...] --raw-artefacts <npz>
        --duration 1200 --hybrid-truth <npz> [--labels ...] [--out <json>]
"""
import argparse
import json
from pathlib import Path

import numpy as np

import hybrid
import waveform
from cleaning import load_clean_units
from compare import MATCH_MIN, MATCH_UM, agreement
from score import eligible


def eligible_units(sort, raw_artefacts, duration, truth):
    by, depth, _ = load_clean_units(sort, raw_artefacts)
    if truth is not None:
        copies = set(hybrid.recovery(sort, truth)['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    units = eligible(sort, by, duration, waveform.screen(sort))
    return {c: np.sort(by[c]) for c in units}, depth


def matched_fraction(a, b):
    """Fraction of a's units with a match (agreement >= MATCH_MIN) in b."""
    (ua, da), (ub, db) = a, b
    order = sorted(ub, key=lambda c: db[c])
    depths = np.array([db[c] for c in order])
    hit = 0
    for c, t in ua.items():
        lo, hi = np.searchsorted(depths, [da[c] - MATCH_UM, da[c] + MATCH_UM + 1e-9])
        hit += any(agreement(t, ub[order[k]]) >= MATCH_MIN for k in range(lo, hi))
    return hit / len(ua)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pair', nargs=2, action='append', required=True)
    p.add_argument('--labels', nargs='*')
    p.add_argument('--raw-artefacts', required=True)
    p.add_argument('--duration', type=float, required=True)
    p.add_argument('--hybrid-truth')
    p.add_argument('--out')
    args = p.parse_args()
    truth = np.load(args.hybrid_truth) if args.hybrid_truth else None
    out = []
    for i, (sa, sb) in enumerate(args.pair):
        a = eligible_units(sa, args.raw_artefacts, args.duration, truth)
        b = eligible_units(sb, args.raw_artefacts, args.duration, truth)
        fa, fb = matched_fraction(a, b), matched_fraction(b, a)
        label = args.labels[i] if args.labels else f'{Path(sa).parent.name} vs {Path(sb).parent.name}'
        out.append({'label': label, 'a': sa, 'b': sb, 'n_a': len(a[0]), 'n_b': len(b[0]),
                    'matched_a_in_b': fa, 'matched_b_in_a': fb})
        print(f'{label:24s} units {len(a[0]):5d} / {len(b[0]):5d}   matched {fa:.1%} / {fb:.1%}', flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
