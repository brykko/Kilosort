"""Unit amplitude measured from the raw data the same way for every sorter, with each unit's contamination.

Amplitude: each unit's mean waveform over up to N_SPIKES of its cleaned spikes (random), on the data
preprocessed as KS4 does before whitening (channel mean and median across channels removed, 300 Hz
high-pass; KS4's BinaryFiltered without whitening or drift correction), on the K_CHAN channels nearest
the unit's median spike position. Amplitude = the largest peak-to-peak over those channels within WINDOW
samples of the spike time, in uV (AP gain 500, +-0.6 V over 10 bits). One pass through each excerpt
serves all its sorts.

Per unit also: spikes, firing rate, contamination (plain and local, as comod.py) and eligibility. Seed 1
of each sorter; all units of the cleaned sort, hybrid copies out (as tradeoff.py).
Writes <OUT>/<excerpt>.npz: per sorter g, g_amp (uV), g_n, g_rate, g_c_plain, g_c_local, g_eligible,
g_ids and g_wave (mean waveforms, units x K_CHAN x samples, uV).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hybrid  # noqa: E402
from cleaning import load_clean_units  # noqa: E402
from comod import unit_estimates  # noqa: E402
from kilosort.io import BinaryFiltered  # noqa: E402
from kilosort.preprocessing import get_highpass_filter  # noqa: E402
from trace_units import MEC_ELIG, SORTS  # noqa: E402

K = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
M = Path('C:/data/ks4_tuning/25843_roger_2019-09-11')
BIN = {'kobe': K / 'search/hybrid/ap_1668-2868s_hybrid.bin', 'mec': M / 'search/ap_13575-14775s.bin'}
OUT = Path('C:/data/ks4_tuning/exploratory/isolation/amplitude')
FS, N_CHAN_BIN, NT, PAD = 30000, 385, 60000, 61
UV_PER_BIT = 1.2e6 / 1024 / 500
N_SPIKES = 1000
K_CHAN = 24
WINDOW = (-30, 50)
DUR = 1200.0


def sort_units(sort, ex, xc, yc, rng):
    """Units (hybrid copies out) with their spike times, nearest channels and a sample of spikes."""
    base = K if ex == 'kobe' else M
    by, _, iv = load_clean_units(sort, base / 'search/raw_artefacts_car.npz')
    if ex == 'kobe':
        copies = set(hybrid.recovery(sort, np.load(base / 'search/hybrid/hybrid_truth.npz'))['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    by = {c: np.sort(t) for c, t in by.items() if t.size}
    T = DUR - float(np.sum(iv[:, 1] - iv[:, 0])) if iv.size else DUR
    clu = np.load(sort / 'spike_clusters.npy').ravel()
    pos = np.load(sort / 'spike_positions.npy')
    o = np.argsort(clu, kind='stable')
    ids, start = np.unique(clu[o], return_index=True)
    xy = {int(c): np.median(p, axis=0) for c, p in zip(ids, np.split(pos[o], start[1:]))}
    if ex == 'kobe':
        elig = {int(c) for c in json.loads((sort / 'evaluation.json').read_text())['S_i']}
    else:
        elig = set(MEC_ELIG[sort.as_posix()])
    ids = sorted(by)
    chans = np.stack([np.argsort(np.hypot(xc - xy[c][0], yc - xy[c][1]))[:K_CHAN] for c in ids])
    samples = [rng.choice(by[c], min(N_SPIKES, by[c].size), replace=False) for c in ids]
    est = [unit_estimates(by[c], T) for c in ids]
    return {'ids': np.array(ids), 'chans': chans, 'samples': samples, 'eligible': np.array([c in elig for c in ids]),
            'n': np.array([e[0] for e in est]), 'rate': np.array([e[1] for e in est]),
            'c_plain': np.array([e[2]['plain'] for e in est]), 'c_local': np.array([e[2]['local'] for e in est])}


def mean_waveforms(ex, units, chan_map, device):
    """One pass through the excerpt; mean waveform of every unit of every sort on its channels."""
    groups = list(units)
    off = np.cumsum([0] + [len(units[g]['ids']) for g in groups])
    st = np.concatenate([np.round(s * FS).astype(np.int64) for g in groups for s in units[g]['samples']])
    uid = np.concatenate([np.full(len(s), off[k] + i) for k, g in enumerate(groups) for i, s in enumerate(units[g]['samples'])])
    chans = torch.from_numpy(np.concatenate([units[g]['chans'] for g in groups])).to(device)
    o = np.argsort(st, kind='stable')
    st, uid = st[o], uid[o]
    lags = torch.arange(*WINDOW, device=device)
    sums = torch.zeros((off[-1], K_CHAN, lags.numel()), device=device)
    counts = torch.zeros(off[-1], device=device)
    bf = BinaryFiltered(str(BIN[ex]), N_CHAN_BIN, FS, NT, PAD, 20, chan_map=chan_map, device=device, do_CAR=True,
                        hp_filter=get_highpass_filter(FS, 300, device))
    edges = np.searchsorted(st, np.arange(bf.n_batches + 1) * NT)
    for b in range(bf.n_batches):
        lo, hi = edges[b], edges[b + 1]
        if lo == hi:
            continue
        X = bf.padded_batch_to_torch(b)
        col = torch.from_numpy(st[lo:hi] - (b * NT - PAD)).to(device)
        u = torch.from_numpy(uid[lo:hi]).to(device)
        snip = X[chans[u][:, :, None], (col[:, None] + lags[None, :])[:, None, :]]
        sums.index_add_(0, u, snip)
        counts.index_add_(0, u, torch.ones_like(u, dtype=torch.float32))
        if b % 100 == 0:
            print(f'  {ex}: batch {b} of {bf.n_batches}', flush=True)
    wave = (sums / counts.clamp(min=1)[:, None, None]).cpu().numpy() * UV_PER_BIT
    return {g: wave[off[k]:off[k + 1]] for k, g in enumerate(groups)}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    rng = np.random.default_rng(0)
    for ex in ('kobe', 'mec'):
        probe = json.loads(((K if ex == 'kobe' else M) / 'search/probe_3a_opt3.json').read_text())
        chan_map, xc, yc = (np.array(probe[k]) for k in ('chanMap', 'xc', 'yc'))
        units = {g: sort_units(SORTS[ex][g], ex, xc, yc, rng) for g in ('default', 'ks25', 'H8')}
        print(f'{ex}: ' + ', '.join(f'{g} {len(u["ids"])} units' for g, u in units.items()), flush=True)
        waves = mean_waveforms(ex, units, chan_map, device)
        out = {}
        for g, u in units.items():
            w = waves[g]
            ptp = w.max(2) - w.min(2)
            amp = ptp.max(1)
            best = w[np.arange(len(w)), ptp.argmax(1)]
            lag = np.arange(*WINDOW)[best.argmin(1)]
            print(f'  {g}: median amplitude {np.median(amp):.0f} uV; trough lag on the peak channel, median {np.median(lag):+.0f} '
                  f'samples (IQR {np.percentile(lag, 25):+.0f} to {np.percentile(lag, 75):+.0f})', flush=True)
            out.update({f'{g}_amp': amp, f'{g}_wave': w.astype(np.float32),
                        **{f'{g}_{k}': u[k] for k in ('ids', 'n', 'rate', 'c_plain', 'c_local', 'eligible')}})
        np.savez(OUT / f'{ex}.npz', **out)
