"""Write the KS4 probe file for an extracted excerpt, from its SpikeGLX AP metadata.

Run with the npx-load environment and the neuropixels `python` folder on PYTHONPATH.
Channels disabled in the metadata (3A reference sites) are left out of the map.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from npx_load import load_probe_geometry

p = argparse.ArgumentParser()
p.add_argument('--meta-dir', required=True, help='folder holding exactly one *.ap.meta')
p.add_argument('--n-chan-bin', type=int, required=True, help='channels saved in the binary, incl. sync')
p.add_argument('--out', required=True)
args = p.parse_args()

g = load_probe_geometry(args.meta_dir)
chan = np.where(g.channel_enabled)[0]
probe = {
    'chanMap': chan.tolist(),
    'xc': g.channel_positions[chan, 0].astype(float).tolist(),
    'yc': g.channel_positions[chan, 1].astype(float).tolist(),
    'kcoords': np.zeros(chan.size).tolist(),
    'n_chan': args.n_chan_bin,
}
Path(args.out).write_text(json.dumps(probe))
print(f'{g.model}: {chan.size} of {g.n_data_chans} channels mapped, '
      f'excluded {np.where(~g.channel_enabled)[0].tolist()} -> {args.out}')
