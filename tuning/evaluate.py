"""Evaluate one sort against the frozen reference (spec §6-7): cleaning, eligibility, S, S_track, C
and the per-sort quantities behind guards G1 and G3-G8. Writes evaluation.json to the sort folder.
For a sort of the hybrid excerpt (--hybrid-truth), hybrid copies are excluded from eligibility and
the recovery of inserted units is recorded (G4). Comparison against default (thresholds, matched units) is compare.py's job.

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

SPLIT_DIST_UM = 50.0
SPLIT_SIMILARITY = 0.8
SPLIT_RATIO = 0.25
CENTRE_S = 1.5e-3
FLANK_S = (2e-3, 10e-3)


def _count(ta, tb, lo, hi):
    """Number of (a, b) spike pairs with lo <= t_b - t_a <= hi."""
    return int(np.sum(np.searchsorted(tb, ta + hi, 'right') - np.searchsorted(tb, ta + lo, 'left')))


def suspected_splits(sort, by_cluster, units, screen):
    """§7.3: eligible pairs with peak channels <= 50 um apart, template similarity >= 0.8 and a
    cross-correlogram rate within +-1.5 ms below 0.25 x the rate at 2-10 ms."""
    pos = np.load(Path(sort) / 'channel_positions.npy')
    sim = np.load(Path(sort) / 'similar_templates.npy')
    xy = {c: pos[screen[c][0]['peak_channel']] for c in units}
    pairs = []
    for i, a in enumerate(units):
        for b in units[i + 1:]:
            if np.linalg.norm(xy[a] - xy[b]) > SPLIT_DIST_UM or sim[a, b] < SPLIT_SIMILARITY:
                continue
            ta, tb = np.sort(by_cluster[a]), np.sort(by_cluster[b])
            centre = _count(ta, tb, -CENTRE_S, CENTRE_S) / (2 * CENTRE_S)
            flank = (_count(ta, tb, *FLANK_S) + _count(ta, tb, -FLANK_S[1], -FLANK_S[0])) / (2 * (FLANK_S[1] - FLANK_S[0]))
            if flank > 0 and centre < SPLIT_RATIO * flank:
                pairs.append({'a': a, 'b': b, 'similarity': float(sim[a, b]), 'ratio': centre / flank})
    return pairs


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
    units, near_noise = eligible(sort, by_cluster, duration, screen)
    sc = score_sort(by_cluster, units, np.load(reference), t0, duration, mask, track=True)
    splits = suspected_splits(sort, by_cluster, units, screen)
    return {
        'sort': str(sort), 'reference': str(reference), 'reference_sha256': sha256(reference),
        'raw_artefacts_sha256': sha256(raw_artefacts),                                    # G8
        'n_units': len(by_cluster), 'n_noise_units': sum(not v[1] for v in screen.values()),
        'n_eligible': len(units),                                                          # G1
        'S': sc['S'], 'S_track': sc['S_track'], 'C': sc['C'],                              # S, G2, G9
        'n_suspected_splits': len(splits),                                                 # G3
        'S_share_near_noise': sum(sc['S_i'][c] for c in near_noise) / sc['S'],             # G5
        'n_near_noise': len(near_noise),
        'fraction_spikes_removed': cleaning['fraction_spikes_removed'],                    # G6
        'n_flagged_pairs': cleaning['duplicates']['n_flagged'],                            # G6
        'n_flagged_pairs_jittered_null': cleaning['duplicates']['n_flagged_jittered_null'],
        'multiunit_events': cleaning['multiunit_events']['n_events'],
        'masked_time_s': float(np.sum(mask[:, 1] - mask[:, 0])) if mask.size else 0.0,
        'S_i': {str(c): v for c, v in sc['S_i'].items()},                                  # G7
        'S_track_i': {str(c): v for c, v in sc['S_track_i'].items()},
        'suspected_splits': splits,
        'hybrid': rec,                                                                     # G4
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
    brief = {k: v for k, v in ev.items() if k not in ('S_i', 'S_track_i', 'suspected_splits', 'hybrid')}
    if ev['hybrid']:
        brief['hybrid'] = {k: v for k, v in ev['hybrid'].items() if k != 'per_unit'}
    print(json.dumps(brief, indent=2))
