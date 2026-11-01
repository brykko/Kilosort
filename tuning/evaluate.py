"""Evaluate one sort against the frozen reference (spec §6-7): cleaning, eligibility, S, S_track and
the per-sort quantities behind the guards (§7.1). Writes evaluation.json to the sort folder.
For a sort of the hybrid excerpt (--hybrid-truth), hybrid copies are excluded from eligibility and
the recovery of inserted units is recorded. Comparison against default (thresholds, matched units) is compare.py's job.

Runs cleaning.py first if the sort folder has no cleaning.npz.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import hybrid
import waveform
from cleaning import load_clean_units, run_cleaning
from score import eligible, score_sort

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(sort, reference, raw_artefacts, t0, duration, hybrid_truth=None):
    sort = Path(sort)
    if not (sort / 'cleaning.npz').exists():
        run_cleaning(sort, duration)
    cleaning = json.loads((sort / 'cleaning.json').read_text())
    by_cluster, _, mask = load_clean_units(sort, raw_artefacts)
    screen = waveform.screen(sort)
    rec = hybrid.recovery(sort, np.load(hybrid_truth)) if hybrid_truth else None
    if rec:
        by_cluster = {c: t for c, t in by_cluster.items() if c not in set(rec['hybrid_copies'])}
    units = eligible(sort, by_cluster, duration, screen)
    sc = score_sort(by_cluster, units, np.load(reference), t0, duration, mask, track=True)
    return {
        'sort': str(sort), 'reference': str(reference), 'reference_sha256': sha256(reference),
        'raw_artefacts_sha256': sha256(raw_artefacts),                                    # raw mask
        'n_units': len(by_cluster), 'n_noise_units': sum(not v[1] for v in screen.values()),
        'n_eligible': len(units),                                                          # unit count
        'S': sc['S'], 'S_track': sc['S_track'],
        'multiunit_events': cleaning['multiunit_events']['n_events'],
        'masked_time_s': float(np.sum(mask[:, 1] - mask[:, 0])) if mask.size else 0.0,
        'S_i': {str(c): v for c, v in sc['S_i'].items()},                                  # untuned units
        'S_track_i': {str(c): v for c, v in sc['S_track_i'].items()},
        'hybrid': rec,                                                                     # hybrid recovery
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sort', required=True)
    p.add_argument('--reference', required=True, help='frozen theta_ref.npz')
    p.add_argument('--raw-artefacts', required=True)
    p.add_argument('--t0', type=float, required=True)
    p.add_argument('--duration', type=float, required=True)
    p.add_argument('--hybrid-truth', help='hybrid_truth.npz, for sorts of the hybrid excerpt')
    args = p.parse_args()
    ev = evaluate(args.sort, args.reference, args.raw_artefacts, args.t0, args.duration, args.hybrid_truth)
    (Path(args.sort) / 'evaluation.json').write_text(json.dumps(ev, indent=2))
    brief = {k: v for k, v in ev.items() if k not in ('S_i', 'S_track_i', 'hybrid')}
    if ev['hybrid']:
        brief['hybrid'] = {k: v for k, v in ev['hybrid'].items() if k != 'per_unit'}
    print(json.dumps(brief, indent=2))
