"""Phase 0 item 8.10 (spec §8.10): waveform adjudication of duplicate pairs flagged <= 150 um apart.

For each such pair (a, b) of a sort, at up to N_EVENTS coincident events (spikes of both within the
flagged lag window, from KS4's output before cleaning) the two un-whitened templates are fitted jointly
(least squares, amplitudes only) to the referenced, high-passed data on the channels where either
template is large. Each unit's usual amplitude is the same fit of its template alone at N_EVENTS of its
non-coincident spikes. The pair is a duplicate if the smaller unit's median conditional amplitude at
coincident events is < 0.3 x its usual amplitude (the larger unit's template already explains the
spike). Reports the fraction of flagged pairs judged duplicates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import waveform

N_EVENTS = 30
NT0 = 20          # KS4 template sample aligned to the spike time
PAD = 300
CHAN_FRAC = 0.2
DUP = 0.3
HALF_W = 2

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True)
p.add_argument('--data', required=True)
p.add_argument('--probe', required=True)
p.add_argument('--n-chan-bin', type=int, required=True)
p.add_argument('--out', required=True)
args = p.parse_args()
rng = np.random.default_rng(0)
sort = Path(args.sort)

chanmap = np.asarray(json.loads(Path(args.probe).read_text())['chanMap'])
raw = np.memmap(args.data, np.int16, 'r').reshape(-1, args.n_chan_bin)
sos = butter(3, 300, 'highpass', fs=30000, output='sos')
W, _ = waveform.load_templates(sort)
nt = W.shape[1]
st = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
cl = np.load(sort / 'spike_clusters.npy').ravel()
times = {c: np.sort(st[cl == c]) for c in np.unique(cl)}
pairs = [pr for pr in json.loads((sort / 'cleaning.json').read_text())['flagged_pairs'] if pr['class'] == 'same_neuron']


def snippet(t0, length, ch):
    x = raw[t0 - PAD:t0 + length + PAD][:, chanmap].astype(np.float32)
    x -= x.mean(0)
    x -= np.median(x, axis=1, keepdims=True)
    return sosfiltfilt(sos, x[:, ch], axis=0)[PAD:PAD + length]


def fit(t_list, units, ch):
    """Least-squares amplitudes of the units' templates placed at t_list, in one window."""
    start = min(t_list) - NT0
    length = max(t_list) - NT0 + nt - start
    y = snippet(start, length, ch).ravel()
    X = np.zeros((length, ch.size, len(units)))
    for j, (t, u) in enumerate(zip(t_list, units)):
        X[t - NT0 - start:t - NT0 - start + nt, :, j] = W[u][:, ch]
    return np.linalg.lstsq(X.reshape(-1, len(units)), y, rcond=None)[0]


results = []
for pr in pairs:
    a, b, lag = pr['a'], pr['b'], pr['lag_samples']
    if times[a].size > times[b].size:
        a, b, lag = b, a, -lag                     # a is the smaller unit; lag = t_b - t_a
    ptp = lambda u: np.ptp(W[u], axis=0)
    ch = np.where((ptp(a) >= CHAN_FRAC * ptp(a).max()) | (ptp(b) >= CHAN_FRAC * ptp(b).max()))[0]
    ta, tb = times[a], times[b]
    j = np.searchsorted(tb, ta + lag - HALF_W)
    hit = (j < tb.size) & (np.abs(tb[np.minimum(j, tb.size - 1)] - (ta + lag)) <= HALF_W)
    ok = (ta > 2 * PAD) & (ta < raw.shape[0] - 2 * PAD)
    coinc = np.where(hit & ok)[0]
    solo = np.where(~hit & ok)[0]
    if coinc.size < 5 or solo.size < 5:
        continue
    cond = [fit([ta[i], tb[j[i]]], [a, b], ch)[0] for i in rng.choice(coinc, min(N_EVENTS, coinc.size), replace=False)]
    usual = [fit([ta[i]], [a], ch)[0] for i in rng.choice(solo, min(N_EVENTS, solo.size), replace=False)]
    ratio = float(np.median(cond) / np.median(usual))
    results.append({'a': int(a), 'b': int(b), 'lag': int(lag), 'distance_um': pr['distance_um'],
                    'n_coincident': int(coinc.size), 'conditional_ratio': ratio, 'duplicate': ratio < DUP})
    print(f"pair {a:4d}/{b:4d} lag {lag:+3d} coincident {coinc.size:6d}  conditional/usual {ratio:+.2f}  "
          f"{'duplicate' if ratio < DUP else 'two neurons'}", flush=True)

n_dup = sum(r['duplicate'] for r in results)
print(f'\n{n_dup} of {len(results)} flagged near pairs judged duplicates by waveform fit '
      f'({100 * n_dup / max(len(results), 1):.0f}% agreement with the coincidence test)')
Path(args.out).write_text(json.dumps({'pairs': results, 'n_duplicate': n_dup, 'n': len(results)}, indent=2))
