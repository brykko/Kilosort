"""Synthetic whitened-scale recordings with known truth, for the round-based vs interleaved simulation.

See notes/simulation.md for the data model. Everything is deterministic given (seed, variant, overrides).
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch

FS = 15000.0
NT = 31
NT0 = 10          # trough sample in the window
NOISE_L = 12.0    # microns
NOISE_SD = 0.8    # Gaussian part; hash and spikes bring the total to about 1, as in whitened data
KS_DIR = r'C:\code\github\Kilosort\kilosort'


def universal_bases():
    """KS4's shipped wTEMP and wPCA, decimated 2:1 to 15 kHz; rows unit norm / orthonormal."""
    d = np.load(os.path.join(KS_DIR, 'wTEMP.npz'))
    wT = d['wTEMP'][:, ::2].astype('float32')
    wT = wT / np.sqrt((wT ** 2).sum(1, keepdims=True))
    wP = d['wPCA'][:, ::2].astype('float64')
    q, _ = np.linalg.qr(wP.T)
    return wT, q.T.astype('float32')


def probe():
    rows = 16
    yc = np.repeat(np.arange(rows) * 20.0, 2)
    xc = np.tile([0.0, 32.0], rows)
    return xc, yc


def waveform(t_ms, kind, p):
    """Unit-peak temporal shape; t_ms relative to the main deflection."""
    g = lambda c, s: np.exp(-0.5 * ((t_ms - c) / s) ** 2)
    if kind == 'regular':
        return -g(0, p['s1']) + p['a'] * g(p['d'], p['s2'])
    if kind == 'triphasic':
        return p['b'] * g(-p['d0'], p['s0']) - g(0, p['s1']) + p['c'] * g(p['d'], p['s2'])
    if kind == 'axonal':      # positive-led, narrow
        return g(0, p['s1']) - p['b'] * g(p['d'], p['s0']) + p['c'] * g(2 * p['d'], p['s2'])
    if kind == 'wide':
        return -g(0, p['s1']) + p['a'] * g(p['d'], p['s2'])
    raise ValueError(kind)


def draw_shape(rng, poor):
    if not poor:
        return 'regular', dict(s1=rng.uniform(.09, .17), a=rng.uniform(.15, .4), d=rng.uniform(.35, .7),
                               s2=rng.uniform(.25, .5))
    kind = rng.choice(['triphasic', 'axonal', 'wide'])
    if kind == 'triphasic':
        return kind, dict(b=rng.uniform(.6, 1.0), d0=rng.uniform(.3, .42), s0=rng.uniform(.1, .16),
                          s1=rng.uniform(.08, .12), c=rng.uniform(.5, .9), d=rng.uniform(.3, .45),
                          s2=rng.uniform(.12, .22))
    if kind == 'axonal':
        return kind, dict(s1=rng.uniform(.08, .12), b=rng.uniform(.7, 1.0), d=rng.uniform(.2, .3),
                          s0=rng.uniform(.08, .12), c=rng.uniform(.3, .6), s2=rng.uniform(.1, .2))
    return kind, dict(s1=rng.uniform(.35, .5), a=rng.uniform(.3, .5), d=rng.uniform(.9, 1.2),
                      s2=rng.uniform(.3, .5))


def jitter_shape(rng, p, sd=0.06):
    return {k: v * (1 + sd * rng.standard_normal()) for k, v in p.items()}


def render(kind, p, x, y, lam, delay, norm, xc, yc, frac=0.0, dy=0.0):
    """(n_chan, NT) template; frac in samples added to spike time; dy rigid shift in microns."""
    t = (np.arange(NT) - NT0 - frac) / FS * 1000.0
    r = np.sqrt((xc - x) ** 2 + (yc - (y + dy)) ** 2)
    amp = np.exp(-r / lam)
    out = np.stack([a * waveform(t - delay * ri, kind, p) for a, ri in zip(amp, r)], 0)
    return out


@dataclass
class Dataset:
    X: torch.Tensor                 # (n_chan, n_samples), float32
    xc: np.ndarray
    yc: np.ndarray
    batch_len: int
    n_batches: int
    t: np.ndarray                   # true trough sample of each spike
    u: np.ndarray                   # true unit
    amp: np.ndarray
    T_true: np.ndarray              # (n_units, n_chan, NT): mean waveform at mean amplitude
    info: dict = field(default_factory=dict)
    pad: int = 2 * NT

    def batch(self, b):
        s = b * self.batch_len
        e = s + self.batch_len
        lo = max(0, s - self.pad)
        hi = min(self.X.shape[1], e + self.pad)
        return self.X[:, lo:hi], lo, s, e


DEFAULTS = dict(n_units=48, T=180.0, batch_s=2.0, rate_med=5.0, rate_sig=1.2, poor_frac=0.25, n_pairs=6,
                n_hash=120, hash_rate=3.0, norm_med=18.0, norm_sig=0.5, amp_sig=0.15, drift_pp=30.0,
                f_sparse=0.10, f_fast=0.06, f_silent=0.35, f_burst=0.20)


def spike_train(rng, rate, T, ref=0.002):
    isi_mean = max(1.0 / rate - ref, 1e-4)
    n = int(rate * T * 1.3 + 30)
    t = np.cumsum(ref + rng.exponential(isi_mean, n))
    t = t[t < T]
    return t


def make_dataset(seed, variant='stationary', **over):
    par = dict(DEFAULTS); par.update(over)
    rng = np.random.default_rng(seed * 1009 + {'stationary': 0, 'drift': 1, 'nonstat': 2}[variant] * 7)
    xc, yc = probe()
    nc = len(xc)
    n_samples = int(par['T'] * FS)
    nu = par['n_units']

    # --- units
    rates = np.clip(np.exp(np.log(par['rate_med']) + par['rate_sig'] * rng.standard_normal(nu)), 0.1, 40.0)
    ord_ = rng.permutation(nu)
    n_sp = int(round(par['f_sparse'] * nu)); n_fast = int(round(par['f_fast'] * nu))
    rates[ord_[:n_sp]] = rng.uniform(0.1, 0.5, n_sp)
    rates[ord_[n_sp:n_sp + n_fast]] = 30.0 * par['rate_med'] / 5.0
    is_poor = rng.random(nu) < par['poor_frac']
    shapes = [draw_shape(rng, is_poor[i]) for i in range(nu)]
    x = rng.uniform(-10, 42, nu)
    y = rng.uniform(5, 295, nu)
    lam = rng.uniform(10, 26, nu)
    delay = rng.uniform(0, 0.002, nu)          # ms per micron
    norm = np.clip(np.exp(np.log(par['norm_med']) + par['norm_sig'] * rng.standard_normal(nu)), 8, 60)

    # near-identical pairs: partners are the units after the first 2*n_pairs positions
    pairs = []
    perm = rng.permutation(nu)
    for i in range(par['n_pairs']):
        a, b = perm[2 * i], perm[2 * i + 1]
        th = rng.uniform(0, 2 * np.pi); dd = rng.uniform(8, 22)
        x[b] = np.clip(x[a] + dd * np.cos(th), -15, 47); y[b] = np.clip(y[a] + dd * np.sin(th), 2, 298)
        lam[b] = lam[a] * rng.uniform(.95, 1.05); delay[b] = delay[a]
        norm[b] = norm[a] * rng.uniform(.87, 1.15)
        is_poor[b] = is_poor[a]
        shapes[b] = (shapes[a][0], jitter_shape(rng, shapes[a][1]))
        pairs.append((int(a), int(b)))

    # --- drift
    nb = int(np.floor(par['T'] / par['batch_s']))
    tgrid = np.linspace(0, 1, 41)
    if variant == 'drift':
        phase = rng.uniform(0, 2 * np.pi)
        dr = np.sin(2 * np.pi * (0.8 + 0.6 * rng.random()) * tgrid + phase)
        dr += 0.5 * np.sin(2 * np.pi * 2.3 * tgrid + rng.uniform(0, 6.28))
        dr = (dr - dr.min()) / (dr.max() - dr.min())
        dr = (dr - 0.5) * par['drift_pp']
    else:
        dr = np.zeros_like(tgrid)
    dy_grid = np.linspace(-par['drift_pp'] / 2, par['drift_pp'] / 2, 11) if variant == 'drift' else np.zeros(1)

    # --- templates: (unit, frac, drift-bin, chan, time)
    fracs = (np.arange(4) - 1.5) / 8.0      # +-0.25 sample at 15 kHz = +-0.5 sample at KS4 30 kHz
    Tmpl = np.zeros((nu, 4, len(dy_grid), nc, NT), 'float32')
    for i in range(nu):
        ref = render(shapes[i][0], shapes[i][1], x[i], y[i], lam[i], delay[i], norm[i], xc, yc)
        scale = norm[i] / np.sqrt((ref ** 2).sum())
        for f, fr in enumerate(fracs):
            for d, dy in enumerate(dy_grid):
                Tmpl[i, f, d] = scale * render(shapes[i][0], shapes[i][1], x[i], y[i], lam[i], delay[i],
                                               norm[i], xc, yc, frac=fr, dy=dy)

    # --- spikes
    all_t, all_u, all_a, all_f, all_d = [], [], [], [], []
    silent = np.zeros(nu, bool); burst = np.zeros(nu, bool)
    if variant == 'nonstat':
        silent = rng.random(nu) < par['f_silent']
        burst = rng.random(nu) < par['f_burst']
    silent_ep = {}
    for i in range(nu):
        r = rates[i] * (0.55 if burst[i] else 1.0)
        ts = spike_train(rng, r, par['T'])
        am = np.exp(par['amp_sig'] * rng.standard_normal(len(ts)) - par['amp_sig'] ** 2 / 2)
        if burst[i] and len(ts):
            extra_t, extra_a = [ts], [am]
            starts = rng.random(len(ts)) < 0.4
            for j in np.flatnonzero(starts):
                n = rng.integers(1, 5); tt = ts[j]; aa = am[j]
                ex_t = []; ex_a = []
                for k in range(n):
                    tt = tt + rng.uniform(.0035, .0075); aa = aa * 0.85
                    ex_t.append(tt); ex_a.append(aa)
                extra_t.append(np.array(ex_t)); extra_a.append(np.array(ex_a))
            ts = np.concatenate(extra_t); am = np.concatenate(extra_a)
            o = np.argsort(ts); ts = ts[o]; am = am[o]
            keep = np.concatenate(([True], np.diff(ts) > 0.002)); ts = ts[keep]; am = am[keep]
        if variant == 'nonstat':
            # slow amplitude change, up to +-15%
            am = am * (1 + 0.15 * rng.uniform(-1, 1) * np.sin(2 * np.pi * ts / par['T'] + rng.uniform(0, 6.28)))
        if silent[i]:
            dur = rng.uniform(.3, .5) * par['T']
            st = rng.uniform(0, par['T'] - dur)
            m = (ts < st) | (ts > st + dur)
            ts = ts[m]; am = am[m]
            silent_ep[i] = (st, st + dur)
        s = ts * FS
        ti = np.round(s).astype(np.int64)
        f = rng.integers(0, 4, len(ti))
        keep = (ti > NT) & (ti < n_samples - 2 * NT)
        ti, f, am, ts = ti[keep], f[keep], am[keep], ts[keep]
        db = np.interp(ts / par['T'], tgrid, dr)
        d = np.abs(db[:, None] - dy_grid[None, :]).argmin(1)
        all_t.append(ti); all_u.append(np.full(len(ti), i)); all_a.append(am); all_f.append(f); all_d.append(d)
    t = np.concatenate(all_t); u = np.concatenate(all_u); a = np.concatenate(all_a)
    f = np.concatenate(all_f); d = np.concatenate(all_d)
    o = np.argsort(t); t, u, a, f, d = t[o], u[o], a[o], f[o], d[o]

    # --- noise: full-length, coloured in time and space
    trng = torch.Generator().manual_seed(int(seed) * 7 + 13)
    Z = torch.randn(nc, n_samples, generator=trng, dtype=torch.float32)
    tk = np.arange(-8, 9)
    k = np.exp(-0.5 * (tk / 0.9) ** 2) - 0.35 * np.exp(-0.5 * (tk / 3.0) ** 2) * (0.9 / 3.0) / 0.35 * 0.8
    k = k - k.mean()
    k = torch.tensor(k / np.sqrt((k ** 2).sum()), dtype=torch.float32)
    Z = torch.nn.functional.conv1d(Z.unsqueeze(1), k.view(1, 1, -1), padding=8).squeeze(1)
    ds2 = np.sqrt((xc[:, None] - xc) ** 2 + (yc[:, None] - yc) ** 2)
    L = np.linalg.cholesky(np.exp(-ds2 / NOISE_L) + 1e-6 * np.eye(nc))
    X = NOISE_SD * (torch.tensor(L, dtype=torch.float32) @ Z)
    del Z

    # --- hash: weak unsorted units, no subsample jitter
    hx = rng.uniform(-10, 42, par['n_hash']); hy = rng.uniform(0, 300, par['n_hash'])
    hn = rng.uniform(2.0, 4.0, par['n_hash'])
    flat = X.view(-1)
    tw = torch.arange(NT, dtype=torch.int64) - NT0   # spike time = trough sample
    for i in range(par['n_hash']):
        kind, p = draw_shape(rng, rng.random() < .25)
        T0 = render(kind, p, hx[i], hy[i], rng.uniform(10, 26), 0.001, hn[i], xc, yc)
        T0 = (T0 * hn[i] / np.sqrt((T0 ** 2).sum())).astype('float32')
        ts = (spike_train(rng, par['hash_rate'], par['T']) * FS).astype(np.int64)
        ts = ts[(ts > NT) & (ts < n_samples - 2 * NT)]
        _add_spikes(flat, n_samples, nc, torch.from_numpy(ts), torch.from_numpy(T0)[None],
                    torch.ones(len(ts)), tw)

    # --- signal
    for i in range(nu):
        m = u == i
        if not m.any():
            continue
        wave = torch.from_numpy(Tmpl[i, f[m], d[m]])       # (n, chan, NT)
        am = torch.from_numpy(a[m].astype('float32'))
        _add_spikes(flat, n_samples, nc, torch.from_numpy(t[m]), wave, am, tw)

    # --- truth template, mean over spikes actually fired: (unit, chan, NT)
    T_true = np.zeros((nu, nc, NT), 'float32')
    for i in range(nu):
        m = u == i
        if not m.any():
            continue
        wt = np.bincount(d[m], weights=a[m], minlength=len(dy_grid)) / m.sum()
        T_true[i] = (wt[:, None, None] * Tmpl[i].mean(0)).sum(0)

    info = dict(rates=rates, is_poor_design=is_poor, kinds=[s[0] for s in shapes], pairs=pairs, x=x, y=y, norm=norm,
                silent=silent, burst=burst, silent_ep=silent_ep, drift=dr, seed=seed, variant=variant, par=par,
                n_spikes_unit=np.bincount(u, minlength=nu))
    return Dataset(X=X, xc=xc, yc=yc, batch_len=int(par['batch_s'] * FS), n_batches=nb, t=t, u=u, amp=a,
                   T_true=T_true, info=info)


def _add_spikes(flat, n_samples, nc, ts, wave, amp, tw):
    """flat is X viewed as (nc * n_samples); adds amp[i] * wave[i or 0] at ts[i]; duplicates accumulate."""
    if len(ts) == 0:
        return
    idx = ts[:, None] + tw[None, :]                          # (n, NT)
    ch = torch.arange(nc, dtype=torch.int64) * n_samples
    full = idx[:, None, :] + ch[None, :, None]               # (n, nc, NT)
    v = amp[:, None, None] * wave
    flat.index_add_(0, full.reshape(-1), v.reshape(-1))
