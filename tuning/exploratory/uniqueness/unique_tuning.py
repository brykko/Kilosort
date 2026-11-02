"""Exploratory prototype (not part of the frozen harness): a tuning-based sort score, "locally unique,
strongly tuned units".

Per sort, each eligible unit (cleaned, §6.1-6.4; hybrid copies excluded on the hybrid excerpt) gets maps
of firing rate against tracked head direction (6° bins, 12° circular smoothing) and position (5 cm bins,
5 cm smoothing), on the usable frames of bayes_decode.py (valid tracking, speed >= 5 cm/s, masked frames
dropped or scaled), from the whole excerpt and from each time half. Every unit is assessed on both.

- Strength: mean vector length (MVL) of the direction map and Skaggs spatial information (bits/spike) of
  the position map, each minus its mean over SHUFFLES copies with the unit's spikes circularly shifted by
  >= SHIFT_MIN_S (bias correction only, no significance test). Strongly tuned: MVL >= MVL_MIN or spatial
  information >= SI_MIN (either, or both).
- Similarity per modality: split-half reliability r_ii (half-A map against half-B map); between units
  r_ij = mean of corr(A_i, B_j) and corr(B_i, A_j); noise-corrected rho_ij = r_ij / sqrt(r_ii r_jj), each
  reliability floored at FLOOR (arithmetic only).
- Twins: two units within R_UM whose rho >= rho* on every modality on which either is strongly tuned;
  every unit of a twin pair drops out.
- Scores over the strongly tuned units without a twin: 'count'; 'saturating', sum of r / (r + R0_HZ) with
  r the rate on the usable frames; 'information', sum of information rates (bits/s on the unit's
  strongest strongly tuned modality, cross-validated over the halves and null-corrected as S_i, floored
  at 0).

Per sort this writes a table (one row per unit) and all pairs within PAIR_MAX_UM, so scores can be
recomputed for any rho*, radius or thresholds. Perturbed copies of a sort (splits, merges, drops) and
random-split fragments (calibration of rho for fragments of one cell) are made from the same units.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import bayes_decode as bd  # noqa: E402
import hybrid  # noqa: E402
import score  # noqa: E402
import waveform  # noqa: E402
from cleaning import load_clean_units, load_sort, mask_intervals  # noqa: E402
from theta_reference import eligible_units as count_eligible  # noqa: E402

SHUFFLES = 20
SHIFT_MIN_S = 60.0
MVL_MIN = 0.25
SI_MIN = 0.5
FLOOR = 0.2
R_UM = 100.0
R0_HZ = 1.0
PAIR_MAX_UM = 500.0
RHO_GRID = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, np.inf]
HD_ANG = (np.arange(bd.HD_BINS) + 0.5) * 2 * np.pi / bd.HD_BINS
DATA = {
    'kobe': dict(base=Path('C:/data/ks4_tuning/26648_kobe_2020-08-21'), tracking='search/tracking_1668-2868s.npz',
                 t0=1668.0, truth='search/hybrid/hybrid_truth.npz'),
    'mec': dict(base=Path('C:/data/ks4_tuning/25843_roger_2019-09-11'), tracking='search/tracking_13575-14775s.npz',
                t0=13575.0, truth=None),
}
DUR = 1200.0


# ---------------- units ----------------

def load_units(sort, ex, amplitudes=False):
    """Eligible units: list of spike-time arrays (excerpt s), unit positions (um), and optionally amplitudes."""
    d = DATA[ex]
    raw = d['base'] / 'search/raw_artefacts_car.npz'
    sort = Path(sort)
    if amplitudes:
        by, _, _, amp = load_clean_units(sort, raw, np.load(sort / 'amplitudes.npy').ravel())
    else:
        by, _, _ = load_clean_units(sort, raw)
    if d['truth']:
        copies = set(hybrid.recovery(sort, np.load(d['base'] / d['truth']))['hybrid_copies'])
        by = {c: t for c, t in by.items() if c not in copies}
    ids = score.eligible(sort, by, DUR, waveform.screen(sort))
    _, _, all_ids, xy = load_sort(sort)
    pos = dict(zip(all_ids.tolist(), xy))
    units = {'t': [by[c] for c in ids], 'xy': np.array([pos[c] for c in ids]), 'ids': list(ids)}
    if amplitudes:
        units['amp'] = [amp[c] for c in ids]
    return units, mask_intervals(sort, raw)


# ---------------- maps ----------------

def flat(units):
    T = np.concatenate(units['t'])
    U = np.repeat(np.arange(len(units['t'])), [t.size for t in units['t']])
    return T, U


def frames_of(beh, T):
    return np.searchsorted(beh['edges'], T, 'right') - 1


def counts(var, beh, f, U, N, use):
    B, b = beh['n_bins'][var], beh['bin'][var]
    sel = (f >= 0) & (f < b.size)
    sel[sel] = use[f[sel]]
    C = np.bincount(U[sel] * B + b[f[sel]], minlength=N * B).reshape(N, B).astype(float)
    return C, np.bincount(b[use], beh['d'][use], B)


def rate_map(var, beh, C, occ):
    Cs, Os = bd.smooth(var, C, beh['shape']), bd.smooth(var, occ, beh['shape'])
    return np.divide(Cs, Os[None, :], out=np.zeros_like(Cs), where=Os[None, :] > 1e-12)


def mvl(M):
    s = M.sum(1)
    return np.divide(np.abs(M @ np.exp(1j * HD_ANG)), s, out=np.zeros(len(M)), where=s > 0)


def skaggs(M, occ):
    vis = occ > 0
    p = occ[vis] / occ[vis].sum()
    L = M[:, vis]
    lbar = L @ p
    ratio = np.divide(L, lbar[:, None], out=np.zeros_like(L), where=lbar[:, None] > 0)
    term = np.zeros_like(ratio)
    pos = ratio > 0
    term[pos] = ratio[pos] * np.log2(ratio[pos])
    return term @ p


def strength(beh, T, U, N, rng):
    use = beh['ok']
    f = frames_of(beh, T)
    Ch, oh = counts('hd', beh, f, U, N, use)
    Cp, op = counts('pos', beh, f, U, N, use)
    out = {'rate': Ch.sum(1) / oh.sum(), 'mvl': mvl(rate_map('hd', beh, Ch, oh)),
           'si': skaggs(rate_map('pos', beh, Cp, op), op)}
    bm, bs = np.zeros(N), np.zeros(N)
    for _ in range(SHUFFLES):
        fs = frames_of(beh, np.mod(T + rng.uniform(SHIFT_MIN_S, DUR - SHIFT_MIN_S, N)[U], DUR))
        bm += mvl(rate_map('hd', beh, counts('hd', beh, fs, U, N, use)[0], oh))
        bs += skaggs(rate_map('pos', beh, counts('pos', beh, fs, U, N, use)[0], op), op)
    out['mvl_c'], out['si_c'] = out['mvl'] - bm / SHUFFLES, out['si'] - bs / SHUFFLES
    return out


def _standardise_rows(M):
    c = M - M.mean(1, keepdims=True)
    sd = np.sqrt((c ** 2).mean(1, keepdims=True))
    return np.divide(c, sd, out=np.zeros_like(c), where=sd > 0)


def half_maps(var, beh, f, U, N):
    maps, occs = [], []
    for h in (False, True):
        C, occ = counts(var, beh, f, U, N, beh['ok'] & (beh['second'] == h))
        maps.append(rate_map(var, beh, C, occ))
        occs.append(occ)
    sup = np.ones(maps[0].shape[1], bool) if var == 'hd' else (occs[0] > 0) & (occs[1] > 0)
    return [_standardise_rows(m[:, sup]) for m in maps]


def similarity(var, beh, f, U, N):
    """Noise-corrected similarity matrix rho (N x N) and split-half reliabilities r."""
    A, B = half_maps(var, beh, f, U, N)
    R = A @ B.T / A.shape[1]
    r = np.diag(R).copy()
    den = np.sqrt(np.maximum(r, FLOOR))
    return (R + R.T) / 2 / den[:, None] / den[None, :], r


def cv_info(var, beh, T, U, N):
    """Information rate (bits/s): cross-validated over the halves, null-corrected by rotation (as S_i)."""
    f = frames_of(beh, T)
    valid = (f >= 0) & (f < beh['ok'].size)
    b, d = beh['bin'][var], beh['d']
    gain, null, ttest = np.zeros(N), np.zeros(N), 0.0
    for h in (False, True):
        train = beh['ok'] & (beh['second'] == h)
        test = beh['ok'] & (beh['second'] != h)
        lam, _ = bd.fit(var, beh, T, U, N, train)
        Ctr, otr = counts(var, beh, f, U, N, train)
        lbar = np.maximum(Ctr.sum(1) / otr.sum(), 1e-12)
        lg = np.log2(lam / lbar[:, None])
        frames = np.flatnonzero(test)
        sel = valid.copy()
        sel[sel] = test[f[sel]]
        fs, us = f[sel], U[sel]

        def g(binf):
            occ = np.bincount(binf[frames], d[frames], lam.shape[1])
            return np.bincount(us, lg[us, binf[fs]], N) - (lam @ occ - lbar * d[frames].sum()) / np.log(2)
        gain += g(b)
        for k in range(1, 7):
            rb = b.copy()
            rb[frames] = b[np.roll(frames, int(round(k / 7 * frames.size)))]
            null += g(rb) / 6
        ttest += d[frames].sum()
    return (gain - null) / ttest


def table(units, beh, rng):
    """Per-unit table and pairs within PAIR_MAX_UM: (i, j, distance, rho_hd, rho_pos)."""
    T, U = flat(units)
    N = len(units['t'])
    f = frames_of(beh, T)
    tab = strength(beh, T, U, N, rng)
    tab['I_hd'], tab['I_pos'] = cv_info('hd', beh, T, U, N), cv_info('pos', beh, T, U, N)
    rh, tab['rel_hd'] = similarity('hd', beh, f, U, N)
    rp, tab['rel_pos'] = similarity('pos', beh, f, U, N)
    xy = units['xy']
    D = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
    i, j = np.nonzero(np.triu(D <= PAIR_MAX_UM, 1))
    tab['xy'] = xy
    pairs = {'i': i.astype(np.int32), 'j': j.astype(np.int32), 'd': D[i, j].astype(np.float32),
             'rho_hd': rh[i, j].astype(np.float32), 'rho_pos': rp[i, j].astype(np.float32)}
    return tab, pairs


def fragment_rho(units, beh, tab, rng):
    """Random-split calibration: rho between the two random halves of each strongly tuned unit, per modality."""
    strong = (tab['mvl_c'] >= MVL_MIN) | (tab['si_c'] >= SI_MIN)
    idx = np.flatnonzero(strong)
    frag = {'t': []}
    for k in idx:
        t = units['t'][k]
        part = rng.integers(2, size=t.size)
        frag['t'] += [t[part == 0], t[part == 1]]
    T, U = flat(frag)
    n = len(frag['t'])
    f = frames_of(beh, T)
    out = {'unit': idx}
    for var in ('hd', 'pos'):
        rho, _ = similarity(var, beh, f, U, n)
        out[f'rho_{var}'] = rho[np.arange(0, n, 2), np.arange(1, n, 2)]
    return out


# ---------------- scores ----------------

def scores(tab, pairs, rho_star, r_um=R_UM, mvl_min=MVL_MIN, si_min=SI_MIN):
    shd, spos = tab['mvl_c'] >= mvl_min, tab['si_c'] >= si_min
    strong = shd | spos
    i, j = pairs['i'], pairs['j']
    nh, npos = shd[i] | shd[j], spos[i] | spos[j]
    twin = (pairs['d'] <= r_um) & (nh | npos) & (~nh | (pairs['rho_hd'] >= rho_star)) & (~npos | (pairs['rho_pos'] >= rho_star))
    tw = np.zeros(strong.size, bool)
    tw[i[twin]] = True
    tw[j[twin]] = True
    uniq = strong & ~tw
    info = np.maximum(np.where(shd, tab['I_hd'], -np.inf), np.where(spos, tab['I_pos'], -np.inf))
    info = np.maximum(info, 0)
    r = tab['rate']
    return {'n_eligible': int(strong.size), 'n_strong': int(strong.sum()), 'n_strong_hd': int(shd.sum()),
            'n_strong_pos': int(spos.sum()), 'n_conj': int((shd & spos).sum()), 'n_twin': int((strong & tw).sum()),
            'count': int(uniq.sum()), 'saturating': float((r / (r + R0_HZ))[uniq].sum()), 'information': float(info[uniq].sum())}


# ---------------- perturbations ----------------

def perturb(units, mode, frac, rng):
    """A perturbed copy of the eligible units; fragments must keep >= 100 spikes in each half (§6.4)."""
    n = len(units['t'])
    t, xy = list(units['t']), list(units['xy'])
    amp = list(units.get('amp', [None] * n))
    if mode == 'drop':
        keep = rng.permutation(n)[int(round(frac * n)):]
        return {'t': [t[k] for k in sorted(keep)], 'xy': np.array([xy[k] for k in sorted(keep)])}
    if mode == 'merge':
        D = np.hypot(*(units['xy'][:, None, :] - units['xy'][None, :, :]).transpose(2, 0, 1))
        used, merged = set(), []
        for a in rng.permutation(n):
            if len(used) >= frac * n:
                break
            if a in used:
                continue
            nb = [b for b in np.flatnonzero((D[a] <= R_UM) & (np.arange(n) != a)) if b not in used]
            if nb:
                b = rng.choice(nb)
                used |= {a, b}
                merged.append((a, b))
        out_t = [t[k] for k in range(n) if k not in used]
        out_xy = [xy[k] for k in range(n) if k not in used]
        for a, b in merged:
            out_t.append(np.sort(np.concatenate([t[a], t[b]])))
            out_xy.append((xy[a] * t[a].size + xy[b] * t[b].size) / (t[a].size + t[b].size))
        return {'t': out_t, 'xy': np.array(out_xy)}
    chosen = set(rng.permutation(n)[:int(round(frac * n))].tolist())
    out_t, out_xy = [], []
    for k in range(n):
        if k not in chosen:
            out_t.append(t[k]); out_xy.append(xy[k])
            continue
        if mode == 'random':
            part = rng.integers(2, size=t[k].size)
        elif mode == 'amplitude':
            part = (amp[k] > np.median(amp[k])).astype(int)
        else:
            part = (t[k] >= DUR / 2 + rng.uniform(-DUR / 4, DUR / 4)).astype(int)
        for p in (0, 1):
            out_t.append(t[k][part == p]); out_xy.append(xy[k])
    by = dict(enumerate(out_t))
    keep = count_eligible(by, DUR, None)
    return {'t': [out_t[k] for k in keep], 'xy': np.array([out_xy[k] for k in keep])}


# ---------------- driver ----------------

def run_one(ex, sort, out, rng, calibrate=False, perturbations=()):
    d = DATA[ex]
    units, mask = load_units(sort, ex, amplitudes=bool(perturbations))
    beh = bd.behaviour(dict(np.load(d['base'] / d['tracking'])), d['t0'], DUR, mask)
    tab, pairs = table(units, beh, rng)
    np.savez_compressed(out, **{f'tab_{k}': v for k, v in tab.items()}, **{f'pair_{k}': v for k, v in pairs.items()})
    s = scores(tab, pairs, 0.9)
    print(f"{out.stem:28s} eligible {s['n_eligible']:5d} strong {s['n_strong']:5d} (hd {s['n_strong_hd']}, pos {s['n_strong_pos']}, "
          f"conj {s['n_conj']}) | rho*0.9: count {s['count']:5d} sat {s['saturating']:7.1f} info {s['information']:7.1f}", flush=True)
    if calibrate:
        fr = fragment_rho(units, beh, tab, rng)
        np.savez_compressed(out.with_name(out.stem + '_fragments.npz'), **fr)
    for mode, frac in perturbations:
        pu = perturb(units, mode, frac, rng)
        ptab, ppairs = table(pu, beh, rng)
        po = out.with_name(f'{out.stem}_{mode}{int(frac * 100)}.npz')
        np.savez_compressed(po, **{f'tab_{k}': v for k, v in ptab.items()}, **{f'pair_{k}': v for k, v in ppairs.items()})
        s = scores(ptab, ppairs, 0.9)
        print(f"  {mode:9s} {frac:4.0%}: eligible {s['n_eligible']:5d} strong {s['n_strong']:5d} | rho*0.9: count {s['count']:5d} "
              f"sat {s['saturating']:7.1f} info {s['information']:7.1f}", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--excerpt', choices=list(DATA), required=True)
    p.add_argument('--sort', nargs=2, action='append', metavar=('LABEL', 'FOLDER'), required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--calibrate', action='store_true', help='also random-split fragments (rho calibration)')
    p.add_argument('--perturb', action='store_true', help='also perturbed copies (splits, merges, drops)')
    args = p.parse_args()
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    pert = [(m, f) for m in ('random', 'amplitude', 'time') for f in (0.1, 1.0)] + [('merge', 0.1), ('drop', 0.1)]
    for label, folder in args.sort:
        run_one(args.excerpt, folder, od / f'{label}.npz', rng, args.calibrate, pert if args.perturb else ())
