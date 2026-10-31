"""Save Motive tracking for a time segment as a compact .npz, using npx_load.

Run with the npx-load environment and the neuropixels `python` folder on PYTHONPATH.
The Motive folder must be a local copy (matmot.mtv + t_synced.npy).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from npx_load.tracking import load_tracking
from npx_load.warnings import WarningCollector

p = argparse.ArgumentParser()
p.add_argument('--motive-dir', required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--t1', type=float, required=True)
p.add_argument('--out', required=True)
args = p.parse_args()

warnings = WarningCollector()
r = load_tracking(Path(args.motive_dir), (args.t0, args.t1), warnings, strict=True)
if len(r['rigid_bodies']) != 1:
    raise SystemExit(f"Expected one rigid body, found {len(r['rigid_bodies'])}")
rb = r['rigid_bodies'][0]
keys = ['x', 'y', 'z', 'hd_azimuth', 'hd_elevation', 'error', 'valid', 'interpolated', 'tracked',
        'speed', 'moving_direction', 'valid_speed']
np.savez(args.out, t=r['t'], **{k: rb[k] for k in keys})
info = {
    'time_range_s': [args.t0, args.t1],
    'n_frames': int(len(r['t'])),
    'hd_offset_mode': rb['hd_offset_mode'],
    'hd_offset_info': rb['hd_offset_info'],
    'warnings': warnings.records,
}
Path(args.out).with_suffix('.json').write_text(json.dumps(info, indent=2, default=str))
print(f"{info['n_frames']} frames, valid {np.mean(rb['valid']):.3f}, tracked {np.mean(rb['tracked']):.3f}, "
      f"interpolated {np.mean(rb['interpolated']):.3f}")
print(json.dumps({k: info[k] for k in ('hd_offset_mode', 'hd_offset_info')}, default=str))
print(f'{len(warnings.records)} warnings')
