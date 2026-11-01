"""Waveform screen for eligibility (spec §6.4).

Metrics are computed per unit from KS4's templates (templates.npy, un-whitened with
whitening_mat_inv.npy), following Bombcell's noise rules (default parameters of py_bombcell; KS4
waveforms are 61 samples, baseline window samples 0-10). Bombcell's other classes (non-somatic,
MUA) are not used:

- peak channel: largest peak-to-peak amplitude;
- troughs and peaks on the peak channel: scipy find_peaks with prominence >= 0.2 x the waveform's
  maximum absolute value (peaks searched separately before and after the main trough);
- peak-trough duration: from the largest-magnitude sample to the opposite extremum after it;
- baseline flatness: max |x| in samples 0-10 over max |x|;
- second-peak-to-trough ratio: largest peak after the trough over the trough;
- spatial decay: exponential fit A exp(-k d) of normalised max |amplitude| on the 10 channels
  nearest in depth (within 33 um laterally) against distance d from the peak channel;
- footprint: depth extent of the channels whose peak-to-peak amplitude is >= 20% of the peak
  channel's.

A unit passes the screen if it is not Bombcell noise and its footprint is <= 300 um.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

FS = 30000
PROMINENCE = 0.2
BASELINE = (0, 10)
EDGE = 4                      # Bombcell blanks the first samples of short waveforms
FOOTPRINT_FRAC = 0.2
LATERAL_UM = 33.0
N_DECAY = 10

DEFAULT = dict(max_peaks=2, max_troughs=1, min_dur_us=100, max_dur_us=1150, max_baseline=0.3,
               max_scnd_peak=0.8, min_decay=0.01, max_decay=0.1, footprint_um=300.0)


def load_templates(sort):
    sort = Path(sort)
    W = np.load(sort / 'templates.npy')                   # units x nt x channels, whitened
    Winv = np.load(sort / 'whitening_mat_inv.npy')
    return W @ Winv, np.load(sort / 'channel_positions.npy')


def unit_metrics(w, pos):
    """Bombcell-style metrics of one template w (nt x channels)."""
    ptp = w.max(0) - w.min(0)
    ch = int(np.argmax(ptp))
    x = w[:, ch].astype(float).copy()
    x[:EDGE] = np.nan
    xm = np.nan_to_num(x)
    prom = PROMINENCE * np.nanmax(np.abs(x))
    tr = find_peaks(-xm, prominence=prom)[0]
    out = {'peak_channel': ch, 'n_troughs': int(tr.size)}
    if tr.size == 0:
        tr = np.array([int(np.nanargmin(x))])
    t_loc = int(tr[np.argmax(-xm[tr])])
    trough = -xm[t_loc]
    before = find_peaks(xm[:t_loc], prominence=prom)[0] if t_loc > 3 else np.array([], int)
    after = find_peaks(xm[t_loc:], prominence=prom)[0] if xm.size - t_loc > 3 else np.array([], int)
    p_after = xm[t_loc + after].max() if after.size else 0.0
    out['n_peaks'] = int(before.size + after.size)

    big = int(np.nanargmax(np.abs(x)))
    other = big + (np.argmin(xm[big:]) if xm[big] > 0 else np.argmax(xm[big:]))
    out['duration_us'] = 1e6 * abs(other - big) / FS
    out['baseline'] = float(np.max(np.abs(w[BASELINE[0]:BASELINE[1], ch])) / np.max(np.abs(w[:, ch])))
    out['scnd_peak_to_trough'] = float(p_after / trough) if trough > 0 else np.inf

    near = np.where(np.abs(pos[:, 0] - pos[ch, 0]) <= LATERAL_UM)[0]
    near = near[np.argsort(np.abs(pos[near, 1] - pos[ch, 1]), kind='stable')][:N_DECAY]
    d = np.linalg.norm(pos[near] - pos[ch], axis=1)
    amp = np.abs(w[:, near]).max(0)
    a = amp / amp.max()
    try:
        k = -curve_fit(lambda r, m_, A: A * np.exp(m_ * r), d, a, p0=[-0.1, 1], maxfev=5000)[0][0]
    except RuntimeError:
        k = np.nan
    out['spatial_decay'] = float(k)

    big_ch = ptp >= FOOTPRINT_FRAC * ptp[ch]
    out['footprint_um'] = float(np.ptp(pos[big_ch, 1]))
    return out


def passes(m, p):
    noise = (m['n_peaks'] > p['max_peaks'] or m['n_troughs'] > p['max_troughs']
             or not p['min_dur_us'] <= m['duration_us'] <= p['max_dur_us']
             or m['baseline'] > p['max_baseline'] or m['scnd_peak_to_trough'] > p['max_scnd_peak']
             or not p['min_decay'] <= m['spatial_decay'] <= p['max_decay'])
    return not noise and m['footprint_um'] <= p['footprint_um']


def screen(sort):
    """Per-cluster metrics and pass flag: {cluster: (metrics, passes)}."""
    W, pos = load_templates(sort)
    out = {}
    for c in range(W.shape[0]):
        m = unit_metrics(W[c], pos)
        out[c] = (m, passes(m, DEFAULT))
    return out
