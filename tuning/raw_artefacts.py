"""Raw-data transient events of an excerpt (spec §6.1), computed once, independent of any sort.

One CPU pass over the excerpt after KS4's referencing (subtract each channel's mean, then the median
across channels): high-pass (3rd-order Butterworth, 300 Hz, forward-backward) each of the probe's
channels; a channel is "active" at a sample if |x| > 6 x its MAD at any point in the surrounding
0.2 ms. Samples where >= MIN_COUNT active channels span >= 1 mm are kept as candidates, merged into
events (gaps <= 0.2 ms), with each event's peak channel count and span.

Everything above the loose MIN_COUNT floor is kept for the §8.9 diagnostic; only events reaching
cleaning.RAW_MASK_MIN_CH channels are masked.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, sosfiltfilt

FS = 30000
CHUNK = 3 * FS
PAD = 3000
MAD_K = 6.0
WIN = 6
SPAN_UM = 1000.0
MIN_COUNT = 8

p = argparse.ArgumentParser()
p.add_argument('--data', required=True)
p.add_argument('--probe', required=True)
p.add_argument('--n-chan-bin', type=int, required=True)
p.add_argument('--out', required=True, help='output .npz (a .json summary is written beside it)')
args = p.parse_args()

probe = json.loads(Path(args.probe).read_text())
order = np.argsort(probe['yc'], kind='stable')
chans = np.asarray(probe['chanMap'])[order]           # channels sorted by depth
ys = np.asarray(probe['yc'], float)[order]

raw = np.memmap(args.data, np.int16, 'r').reshape(-1, args.n_chan_bin)
n = raw.shape[0]
hp = butter(3, 300, 'highpass', fs=FS, output='sos')


def chunk(start, stop):
    """Referenced, high-passed block of the depth-ordered channels."""
    a, b = max(start - PAD, 0), min(stop + PAD, n)
    x = raw[a:b][:, chans].astype(np.float32)
    x -= x.mean(0)
    x -= np.median(x, axis=1, keepdims=True)
    x = sosfiltfilt(hp, x, axis=0)
    return x[start - a: start - a + (stop - start)]


# MAD per channel from 20 one-second chunks spread through the excerpt
mad = np.median(np.abs(np.concatenate([chunk(s, s + FS) for s in np.linspace(PAD, n - FS - PAD, 20).astype(int)])), axis=0) / 0.6745

cand_s, cand_count, cand_span = [], [], []
t0 = time.time()
for start in range(0, n, CHUNK):
    stop = min(start + CHUNK, n)
    x = chunk(start, stop)
    active = maximum_filter1d((np.abs(x) > MAD_K * mad).astype(np.uint8), WIN, axis=0).astype(bool)
    count = active.sum(1)
    idx = np.where(count >= MIN_COUNT)[0]
    if idx.size:
        a = active[idx]
        lo = ys[np.argmax(a, axis=1)]
        hi = ys[a.shape[1] - 1 - np.argmax(a[:, ::-1], axis=1)]
        keep = hi - lo >= SPAN_UM
        cand_s.append(idx[keep] + start); cand_count.append(count[idx][keep]); cand_span.append((hi - lo)[keep])
    print(f'\r{100 * stop / n:5.1f}%  {time.time() - t0:5.0f} s', end='', flush=True)
print()

s = np.concatenate(cand_s) if cand_s else np.empty(0, int)
cnt = np.concatenate(cand_count) if cand_count else np.empty(0, int)
spn = np.concatenate(cand_span) if cand_span else np.empty(0)
# merge candidate samples into events (gaps <= WIN samples)
breaks = np.where(np.diff(s) > WIN)[0] + 1
ev = [(g[0], g[-1] + 1, c.max(), sp.max()) for g, c, sp in zip(np.split(s, breaks), np.split(cnt, breaks), np.split(spn, breaks))] if s.size else []
events = np.array(ev, float).reshape(-1, 4)
np.savez(args.out, mad=mad, channels_by_depth=chans, y_by_depth=ys, events=events)
summary = {
    'n_samples': int(n), 'runtime_s': time.time() - t0,
    'transient_events_by_min_channels': {str(k): int(np.sum(events[:, 2] >= k)) for k in (8, 16, 32, 64, 128, 256)},
}
Path(args.out).with_suffix('.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
