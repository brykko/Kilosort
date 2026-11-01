"""Hybrid ground truth (spec §7.4): insert 40 time-shifted copies of reference-sort units into a copy of
the excerpt.

- Sources: well-isolated reference-sort units (eligible under §6.4, refractory contamination <= 5%,
  in no flagged duplicate pair or suspected split pair, >= 600 spikes after cleaning). 5 pairs of HD
  cells (mean vector length of the tracked-heading tuning >= 0.25, preferred directions >= 90° apart)
  and 30 single units, one from each amplitude quantile of the remaining isolated units.
- Waveforms: mean of 300 spikes on all channels after KS4-style referencing (channel mean, then
  median across channels) and a 300 Hz high-pass, samples -30..+59 around the spike time, kept on
  channels within 120 um of the peak channel and tapered at both ends.
- Placement: moved along the probe by a multiple of 40 um (the column pattern's period) and >= 200 um
  from the source, with the footprint inside the probe (reference sites, which record nothing, simply
  get no signal), and >= 60 um from other inserted units; the two members of an HD pair are <= 60 um
  apart.
- Spike trains: the source unit's cleaned spike train circularly shifted by a fixed offset in
  [60, duration - 60] s (shared by the two members of a pair), each spike scaled by an amplitude drawn
  from the source unit's KS4 amplitudes (divided by their median).
- Drift: each spike is placed at its target depth minus the reference sort's drift estimate for its
  batch (KS4 corrects by reading the data at y - dshift), by linear interpolation along each column.

Writes the hybrid excerpt, hybrid_truth.npz (per inserted spike: sample, unit; per unit: source, shift,
depths, pair) and hybrid.json (selection, placement, SHA-256 of the hybrid file).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

import waveform
from cleaning import load_clean_units
from evaluate import suspected_splits
from score import eligible
from theta_reference import CENTRES, cycle_counts, tuning

FS = 30000
PRE, POST = 30, 60
SNIP_PAD = 300
N_WF_SPIKES = 300
FOOT_UM = 120.0
PERIOD_UM = 40.0
MIN_MOVE_UM, MIN_GAP_UM, PAIR_UM = 200.0, 60.0, 60.0
N_PAIRS, N_SINGLE = 5, 30
HD_MVL, PAIR_DPD = 0.25, np.pi / 2
MAX_CONTAM, MIN_SPIKES = 0.05, 600
CHUNK = 3 * FS

p = argparse.ArgumentParser()
p.add_argument('--sort', required=True, help='reference sort folder')
p.add_argument('--reference', required=True, help="its theta_ref.npz (cycles and tracked heading)")
p.add_argument('--data', required=True)
p.add_argument('--probe', required=True)
p.add_argument('--n-chan-bin', type=int, required=True)
p.add_argument('--raw-artefacts', required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--t0', type=float, required=True)
p.add_argument('--out', required=True, help='output .bin (hybrid_truth.npz and hybrid.json go beside it)')
p.add_argument('--seed', type=int, default=0)
args = p.parse_args()
rng = np.random.default_rng(args.seed)
sort, out = Path(args.sort), Path(args.out)
if out.exists():
    raise SystemExit(f'{out} already exists')

# ---------------- source units ----------------
by_cluster, depth, mask = load_clean_units(sort, args.raw_artefacts)
screen = waveform.screen(sort)
units = eligible(sort, by_cluster, args.duration, screen, max_contamination=0.10)[0]   # v0.6 criteria
flagged = {u for pr in json.loads((sort / 'cleaning.json').read_text())['flagged_pairs'] for u in (pr['a'], pr['b'])}
split = {u for pr in suspected_splits(sort, by_cluster, units, screen) for u in (pr['a'], pr['b'])}


def contamination(t):
    t = np.sort(t)
    return np.sum(np.diff(t) < 1.5e-3) * args.duration / (2 * (1.5e-3 - 0.25e-3) * t.size ** 2)


isolated = [c for c in units if c not in flagged and c not in split
            and by_cluster[c].size >= MIN_SPIKES and contamination(by_cluster[c]) <= MAX_CONTAM]

ref = np.load(args.reference)
bounds = ref['cycle_bounds_s'] - args.t0
hd = ref['hd_tracked']
ok = np.isfinite(hd)
num, occ = tuning(np.where(ok, hd, 0.0), cycle_counts(by_cluster, isolated, bounds), np.diff(bounds), ok)
rate = num / np.maximum(occ, 1e-9)[None, :]          # tuning curves against tracked heading (60 bins)
z = rate @ np.exp(1j * CENTRES) / rate.sum(1)
mvl, pdir = np.abs(z), np.angle(z)
hd_cells = sorted((c for c, m in zip(isolated, mvl) if m >= HD_MVL), key=lambda c: -mvl[isolated.index(c)])
pd_of = dict(zip(isolated, pdir))

pairs, used = [], set()
for a in hd_cells:
    if a in used or len(pairs) == N_PAIRS:
        continue
    for b in hd_cells:
        if b != a and b not in used and abs(np.angle(np.exp(1j * (pd_of[a] - pd_of[b])))) >= PAIR_DPD:
            pairs.append((a, b)); used |= {a, b}
            break
if len(pairs) < N_PAIRS:
    raise SystemExit(f'only {len(pairs)} HD pairs available')

W, pos = waveform.load_templates(sort)
amp_t = {c: np.ptp(W[c][:, screen[c][0]['peak_channel']]) for c in isolated}
rest = sorted((c for c in isolated if c not in used), key=lambda c: amp_t[c])
singles = [int(rng.choice(q)) for q in np.array_split(np.array(rest), N_SINGLE)]
sources = [c for pr in pairs for c in pr] + singles

# ---------------- waveforms ----------------
probe = json.loads(Path(args.probe).read_text())
chanmap = np.asarray(probe['chanMap'])
xc, yc = np.asarray(probe['xc'], float), np.asarray(probe['yc'], float)
raw = np.memmap(args.data, np.int16, 'r').reshape(-1, args.n_chan_bin)
n_samp = raw.shape[0]
sos = butter(3, 300, 'highpass', fs=FS, output='sos')
st_all = np.load(sort / 'spike_times.npy').ravel()
cl_all = np.load(sort / 'spike_clusters.npy').ravel()
taper = tukey(PRE + POST, 0.3)


def mean_waveform(c):
    t = st_all[cl_all == c]
    t = t[(t > SNIP_PAD + PRE) & (t < n_samp - SNIP_PAD - POST)]
    acc = np.zeros((PRE + POST, chanmap.size))
    for s in rng.choice(t, min(N_WF_SPIKES, t.size), replace=False):
        x = raw[s - SNIP_PAD:s + SNIP_PAD][:, chanmap].astype(np.float32)
        x -= x.mean(0)
        x -= np.median(x, axis=1, keepdims=True)
        acc += sosfiltfilt(sos, x, axis=0)[SNIP_PAD - PRE:SNIP_PAD + POST]
    wf = acc / min(N_WF_SPIKES, t.size)
    peak = int(np.argmax(np.ptp(wf, axis=0)))
    wf[:, np.abs(yc - yc[peak]) > FOOT_UM] = 0
    return wf * taper[:, None], peak


t_start = time.time()
wfs, peaks = {}, {}
for c in sources:
    wfs[c], peaks[c] = mean_waveform(c)
print(f'waveforms: {time.time() - t_start:.0f} s')

# ---------------- placement ----------------
def footprint_ok(c, move):
    nz = np.where(np.any(wfs[c] != 0, axis=0))[0]
    return yc[nz].min() + move >= yc.min() and yc[nz].max() + move <= yc.max()


def moves(c):
    m = np.arange(-120, 121) * PERIOD_UM
    m = m[np.abs(m) >= MIN_MOVE_UM]
    return [mv for mv in rng.permutation(m) if footprint_ok(c, mv)]


placed = {}  # unit -> move (um)


def far_from_others(y, exclude=()):
    return all(abs(y - (yc[peaks[o]] + mv)) >= MIN_GAP_UM for o, mv in placed.items() if o not in exclude)


for a, b in pairs:
    for ma in moves(a):
        ya = yc[peaks[a]] + ma
        if not far_from_others(ya):
            continue
        mb = next((mv for mv in moves(b) if abs(yc[peaks[b]] + mv - ya) <= PAIR_UM
                   and far_from_others(yc[peaks[b]] + mv)), None)
        if mb is not None:
            placed[a], placed[b] = float(ma), float(mb)
            break
    else:
        raise SystemExit(f'could not place pair {a}, {b}')
for c in singles:
    mv = next((mv for mv in moves(c) if far_from_others(yc[peaks[c]] + mv)), None)
    if mv is None:
        raise SystemExit(f'could not place unit {c}')
    placed[c] = float(mv)

# ---------------- spike trains ----------------
amps = np.load(sort / 'amplitudes.npy').ravel()
shift_s = {}
for a, b in pairs:
    shift_s[a] = shift_s[b] = float(rng.uniform(60, args.duration - 60))
for c in singles:
    shift_s[c] = float(rng.uniform(60, args.duration - 60))
spk_t, spk_u, spk_a = [], [], []
for k, c in enumerate(sources):
    t = (np.round(by_cluster[c] * FS).astype(np.int64) + int(round(shift_s[c] * FS))) % n_samp
    t = np.sort(t[(t >= SNIP_PAD) & (t < n_samp - SNIP_PAD)])
    a = amps[cl_all == c]
    spk_t.append(t); spk_u.append(np.full(t.size, k)); spk_a.append(rng.choice(a / np.median(a), t.size))
spk_t, spk_u, spk_a = (np.concatenate(v) for v in (spk_t, spk_u, spk_a))
o = np.argsort(spk_t, kind='stable')
spk_t, spk_u, spk_a = spk_t[o], spk_u[o], spk_a[o]

# ---------------- drifted waveforms ----------------
ops = np.load(sort / 'ops.npy', allow_pickle=True).item()
dshift = np.asarray(ops['dshift']).ravel()
offset_um = -dshift[np.minimum(spk_t // ops['batch_size'], dshift.size - 1)]   # raw position = y - dshift
offsets = np.unique(offset_um)
cols = {x: np.where(xc == x)[0][np.argsort(yc[xc == x])] for x in np.unique(xc)}


def placed_waveform(c, move, off):
    """Waveform on the probe's channels for unit c moved by move + off um (linear along columns)."""
    out_w = np.zeros_like(wfs[c])
    for idx in cols.values():
        ys = yc[idx]
        src = ys - move - off              # where each target channel samples the source waveform
        j = np.where((src >= ys[0]) & (src <= ys[-1]))[0]
        k = np.clip(np.searchsorted(ys, src[j]), 1, ys.size - 1)
        w = (src[j] - ys[k - 1]) / (ys[k] - ys[k - 1])
        col = wfs[c][:, idx]
        out_w[:, idx[j]] = col[:, k - 1] * (1 - w) + col[:, k] * w
    return out_w


bank = {(k, off): placed_waveform(c, placed[c], off) for k, c in enumerate(sources) for off in offsets}
spk_off = np.searchsorted(offsets, offset_um)

# ---------------- write ----------------
out.parent.mkdir(parents=True, exist_ok=True)
h = hashlib.sha256()
t_start = time.time()
with open(out, 'wb') as f:
    for start in range(0, n_samp, CHUNK):
        stop = min(start + CHUNK, n_samp)
        x = np.array(raw[start:stop], dtype=np.int32)
        lo, hi = np.searchsorted(spk_t, [start - POST, stop + PRE])
        for t, k, a, oi in zip(spk_t[lo:hi], spk_u[lo:hi], spk_a[lo:hi], spk_off[lo:hi]):
            w = a * bank[(int(k), offsets[oi])]
            s0, s1 = max(t - PRE, start), min(t + POST, stop)
            x[s0 - start:s1 - start, chanmap] += np.round(w[s0 - (t - PRE):s1 - (t - PRE)]).astype(np.int32)
        b = np.clip(x, -32768, 32767).astype(np.int16).tobytes()
        f.write(b); h.update(b)
        print(f'\r{100 * stop / n_samp:5.1f}%  {time.time() - t_start:5.0f} s', end='', flush=True)
print()

y_target = {c: float(yc[peaks[c]] + placed[c]) for c in sources}
np.savez(out.with_name('hybrid_truth.npz'), spike_samples=spk_t, spike_unit=spk_u, spike_scale=spk_a,
         source_cluster=np.array(sources), shift_s=np.array([shift_s[c] for c in sources]),
         source_depth_um=np.array([float(yc[peaks[c]]) for c in sources]),
         target_depth_um=np.array([y_target[c] for c in sources]),
         pair=np.array([i // 2 if i < 2 * N_PAIRS else -1 for i in range(len(sources))]),
         waveforms=np.stack([wfs[c] for c in sources]))
summary = {
    'source_sort': str(sort), 'seed': args.seed, 'data': args.data, 'n_isolated_candidates': len(isolated),
    'n_hd_cells': len(hd_cells),
    'pairs': [{'units': [a, b], 'mvl': [float(mvl[isolated.index(a)]), float(mvl[isolated.index(b)])],
               'pd_deg': [float(np.degrees(pd_of[a])), float(np.degrees(pd_of[b]))],
               'target_depth_um': [y_target[a], y_target[b]], 'shift_s': shift_s[a]} for a, b in pairs],
    'units': [{'source': c, 'n_spikes': int(np.sum(spk_u == k)), 'amplitude_ptp': float(np.ptp(wfs[c])),
               'source_depth_um': float(yc[peaks[c]]), 'target_depth_um': y_target[c], 'shift_s': shift_s[c]}
              for k, c in enumerate(sources)],
    'drift_offsets_um': offsets.tolist(),
    'n_inserted_spikes': int(spk_t.size),
    'hybrid_sha256': h.hexdigest(),
}
out.with_name('hybrid.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'units'}, indent=2))
