"""Build the frozen reference heading theta_ref from a reference sort (spec §5.1).

theta_ref: per-cycle Poisson decoding (posterior mean) with tuning refitted for 3 passes, all eligible
units (§6.4, including the waveform screen), moving theta cycles only, on the sort after cleaning
(spec §6.1-6.3). Prerequisites in the
sort folder: acg_class.npz (tuning/acg_class.py, elv environment) and cleaning.npz
(tuning/cleaning.py). Cycles more than 50% masked get no reference heading; others have their
duration scaled by the unmasked fraction. Writes theta_ref.npz (cycles, moving flags, theta_ref,
units, tracked heading per cycle for S_track) and theta_ref.json (summary, sanity floor against
tracked heading, SHA-256 of the arrays).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cleaning import load_clean_units, masked_fraction
from score import eligible
from theta_reference import compare_to_tracking, cycle_counts, internal_direction, theta_cycles

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True)
p.add_argument('--tracking', required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--raw-artefacts', required=True, help='raw_artefacts npz of the excerpt (spec §6.1)')
args = p.parse_args()

sort = Path(args.sort)
by_cluster, _, mask = load_clean_units(sort, args.raw_artefacts)
acg = np.load(sort / 'acg_class.npz')
source = [int(c) for c, lab in zip(acg['cluster'], acg['raw_acg_label']) if lab in (1, 2)]
bounds, dur, moving, hd = theta_cycles(by_cluster, source, np.load(args.tracking), args.t0, args.duration)
frac = masked_fraction(bounds, mask)
usable = moving & (frac <= 0.5)
units = eligible(sort, by_cluster, args.duration)[0]
theta = internal_direction(cycle_counts(by_cluster, units, bounds), dur * (1 - frac), usable, hd, 'poisson_mean')[-1]
theta[~usable] = np.nan

arrays = {'cycle_bounds_s': bounds + args.t0, 'moving': moving, 'theta_ref': theta,
          'units': np.array(units), 'theta_source_units': np.array(source),
          'hd_tracked': np.where(moving, hd, np.nan)}
h = hashlib.sha256()
for k in sorted(arrays):
    h.update(k.encode()); h.update(np.ascontiguousarray(arrays[k]).tobytes())
np.savez(sort / 'theta_ref.npz', **arrays)
summary = {
    'sort': str(sort), 'variant': 'poisson posterior mean, 3 passes, no temporal prior',
    'n_units': len(units), 'n_theta_source_units': len(source),
    'n_cycles': int(dur.size), 'n_moving_cycles': int(moving.sum()),
    'n_moving_cycles_dropped_masked': int((moving & ~usable).sum()),
    'masked_time_s': float(np.sum(mask[:, 1] - mask[:, 0])) if mask.size else 0.0,
    'cycle_duration_ms_median': float(np.median(dur) * 1e3),
    'sanity_floor_vs_tracking': compare_to_tracking(theta, hd, usable),
    'arrays_sha256': h.hexdigest(),
}
(sort / 'theta_ref.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
