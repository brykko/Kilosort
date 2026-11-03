"""Tables from the cached lag histograms (dup_lags.py): within-unit short-lag profile and duplicate excess,
cross-unit coincidence excess by distance and template similarity; per sorter, replicate and variant.

Within-unit: pooled ratio = sum_u H_u(L) / sum_u E_u with E_u the unit's mean count per 1-sample lag bin at
5-20 ms (alt: 2-5 ms). Duplicate excess = sum over lags of (H_u(L) - F_u) with F_u the unit's mean count per bin
at 1.0-1.5 ms (the refractory floor: contamination plus any early own spikes), in bands 0-0.25 ms (lags 0-7),
0.25-0.5 (8-14), 0.5-0.75 (15-22), 0.75-1.0 (23-29). Hill-window violations: lags 8-44 (0.25 < dt < 1.5 ms).
Units with >= 100 spikes. Cross-unit: H(|lag|) per (distance, similarity) category / mean at 5-10 ms
(lag 0 counted once, so its expected is half a bin).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dup_common import CACHE, DIST_EDGES, SIM_EDGES, SORTS  # noqa: E402

NMIN = 100
BANDS = [(0, 8), (8, 15), (15, 23), (23, 30)]
BN = ['0-.25', '.25-.5', '.5-.75', '.75-1']
PROFILE = [(0, 4), (4, 8), (8, 12), (12, 15), (15, 23), (23, 30), (30, 45), (45, 60), (60, 90)]
PN = ['0-.13', '.13-.27', '.27-.4', '.4-.5', '.5-.75', '.75-1', '1-1.5', '1.5-2', '2-3']
SORTERS = ('ks4_default', 'ks4_H8', 'ks25_std', 'ks25_low')
ND, NS = DIST_EDGES.size - 1, SIM_EDGES.size - 1


def within_stats(Hw, n, sel=None):
    keep = n >= NMIN if sel is None else (n >= NMIN) & sel
    H = Hw[keep].astype(float); nn = n[keep]
    E = H[:, 150:601].mean(1); E2 = H[:, 60:151].mean(1); F = H[:, 30:45].mean(1)
    prof = [H[:, a:b].sum() / (E.sum() * (b - a)) for a, b in PROFILE]
    prof2 = [H[:, a:b].sum() / (E2.sum() * (b - a)) for a, b in PROFILE]
    exc = [(H[:, a:b].sum() - F.sum() * (b - a)) for a, b in BANDS]
    V = H[:, 8:45].sum()
    # per-unit: peak 0.27-0.5 ms over floor 1-1.5 ms, units with enough floor counts
    ok = F * 15 >= 10
    pk = (H[ok, 8:15].mean(1) / np.maximum(F[ok], 1e-9))
    return {'prof': prof, 'prof2': prof2, 'floor': F.sum() / E.sum(), 'exc': exc, 'N': nn.sum(), 'V': V,
            'units': int(keep.sum()), 'pk_q': np.percentile(pk, [25, 50, 75]) if pk.size else [np.nan] * 3,
            'pk_n': int(ok.sum()), 'V0': H[:, 0:8].sum()}


def cross_stats(Hx, N):
    H = Hx.reshape(ND, NS, -1).astype(float)
    base = H[..., 150:301].mean(-1)
    def ratio(h, b, a, z):
        exp = b * (z - a) - (b / 2 if a == 0 else 0)
        return h[..., a:z].sum(-1) / np.maximum(exp, 1e-9), h[..., a:z].sum(-1) - exp
    return H, base, ratio


def load(ex, g, k, v):
    p = CACHE / f'{ex}_{g}_{k}_{v}.npz'
    return np.load(p) if p.exists() else None


def fmt(x, f='{:7.2f}'):
    return ''.join(f.format(v) for v in x)


if __name__ == '__main__':
    for ex in ('kobe', 'mec'):
        print(f'\n================ {ex} ================')
        print('\n(a) within-unit pooled profile, observed/expected from each unit\'s 5-20 ms level, lag bands (ms):')
        print(f"{'sorter':14s}{'var':6s}{'r':>3s}{'units':>6s}" + ''.join(f'{p:>8s}' for p in PN) + '   floor(1-1.5)')
        for g in SORTERS:
            for v in ('pre', 'raw', 'clean'):
                for k in range(1, 6):
                    z = load(ex, g, k, v)
                    if z is None:
                        continue
                    s = within_stats(z['Hw'], z['n'])
                    print(f'{g:14s}{v:6s}{k:3d}{s["units"]:6d}' + fmt(s['prof'], '{:8.3f}') + f'   {s["floor"]:.3f}')
        for st in ('extracted', 'merge3'):
            z = load(ex, 'ks25traced', 1, st)
            s = within_stats(z['Hw'], z['n'])
            print(f'{"ks25traced":14s}{st[:6]:6s}{1:3d}{s["units"]:6d}' + fmt(s['prof'], '{:8.3f}') + f'   {s["floor"]:.3f}')

        print('\n(a2) same, normalised by the 2-5 ms level (seed/replicate 1, raw):')
        for g in SORTERS:
            z = load(ex, g, 1, 'raw'); s = within_stats(z['Hw'], z['n'])
            print(f'{g:14s}' + fmt(s['prof2'], '{:8.3f}'))

        print('\n(b) duplicate excess above the 1.0-1.5 ms floor: pairs per 1000 spikes by band (ms); share of Hill-window'
              ' violations (0.25-1.5 ms) that are excess at 0.25-1 ms; pairs < 0.25 ms per 1000 spikes; per-unit peak'
              ' (0.27-0.5 ms / floor) quartiles')
        print(f"{'sorter':14s}{'var':6s}{'r':>3s}" + ''.join(f'{b:>8s}' for b in BN) + '  sum.25-1  share  V/1000  pk25  pk50  pk75 (units)')
        for g in SORTERS:
            for v in ('pre', 'raw', 'clean'):
                acc = []
                for k in range(1, 6):
                    z = load(ex, g, k, v)
                    if z is None:
                        continue
                    s = within_stats(z['Hw'], z['n'])
                    e = np.array(s['exc']) / s['N'] * 1000
                    sh = sum(s['exc'][1:]) / s['V']
                    acc.append((e, sh))
                    print(f'{g:14s}{v:6s}{k:3d}' + fmt(e, '{:8.2f}') + f'{e[1:].sum():10.2f}{sh:7.1%}{s["V"] / s["N"] * 1000:8.1f}'
                          + fmt(s['pk_q'], '{:6.2f}') + f' ({s["pk_n"]})')
                if len(acc) > 1:
                    e = np.mean([a[0] for a in acc], 0); sh = np.mean([a[1] for a in acc])
                    print(f'{g:14s}{v:6s}{"mn":>3s}' + fmt(e, '{:8.2f}') + f'{e[1:].sum():10.2f}{sh:7.1%}')
        for st in ('extracted', 'merge3'):
            z = load(ex, 'ks25traced', 1, st); s = within_stats(z['Hw'], z['n'])
            e = np.array(s['exc']) / s['N'] * 1000
            print(f'{"ks25traced":14s}{st[:6]:6s}{1:3d}' + fmt(e, '{:8.2f}') + f'{e[1:].sum():10.2f}{sum(s["exc"][1:]) / s["V"]:7.1%}'
                  f'{s["V"] / s["N"] * 1000:8.1f}' + fmt(s['pk_q'], '{:6.2f}') + f' ({s["pk_n"]})')

        print('\n(b2) controls (raw, replicates pooled): excess 0.25-1 ms per 1000 spikes and share of Hill violations,'
              ' by within-sorter amplitude tercile and by firing rate (Hz)')
        RB = [0, 1, 5, 20, np.inf]
        print(f"{'sorter':14s}" + ''.join(f'{f"amp T{i}":>16s}' for i in (1, 2, 3)) + ''.join(f'{f"{a}-{b}Hz":>16s}' for a, b in zip(RB[:-1], RB[1:])))
        for g in SORTERS:
            cells_a = [[0, 0, 0] for _ in range(3)]; cells_r = [[0, 0, 0] for _ in range(4)]
            for k in range(1, 6):
                z = load(ex, g, k, 'raw')
                if z is None:
                    continue
                n, amp, T = z['n'], z['amp'], float(z['T'])
                q = np.nanpercentile(amp[n >= NMIN], [100 / 3, 200 / 3])
                ta = np.digitize(amp, q); tr = np.digitize(n / T, RB[1:-1])
                for i in range(3):
                    s = within_stats(z['Hw'], n, ta == i)
                    cells_a[i] = [cells_a[i][0] + sum(s['exc'][1:]), cells_a[i][1] + s['N'], cells_a[i][2] + s['V']]
                for i in range(4):
                    if np.sum((tr == i) & (n >= NMIN)) == 0:
                        continue
                    s = within_stats(z['Hw'], n, tr == i)
                    cells_r[i] = [cells_r[i][0] + sum(s['exc'][1:]), cells_r[i][1] + s['N'], cells_r[i][2] + s['V']]
            print(f'{g:14s}' + ''.join(f'{1000 * a / max(b, 1):7.2f} ({a / max(c, 1):5.1%})' for a, b, c in cells_a + cells_r))

        print('\n(c) cross-unit coincidence: observed/expected (from 5-10 ms) at |lag| bands, by distance (all similarities);'
              ' and excess pairs (0-1 ms) per 1000 spikes')
        LB = [(0, 8), (8, 15), (15, 30), (30, 60)]
        dn = [f'{int(a)}-{int(b)}' if b < 1e8 else f'>{int(a)}' for a, b in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]
        for g in SORTERS:
            for v in ('pre', 'raw', 'clean'):
                for k in (1,) if v != 'raw' else range(1, 6):
                    z = load(ex, g, k, v)
                    if z is None:
                        continue
                    H, base, ratio = cross_stats(z['Hx'], z['n'].sum())
                    Hd, bd = H.sum(1), base.sum(1)
                    N = z['n'].sum()
                    if k == 1:
                        print(f'  {g} {v} r{k}:  um      0-.25  .25-.5   .5-1    1-2   exc/1000sp(0-1ms)')
                    rows = []
                    for d in range(ND):
                        r = [ratio(Hd[d], bd[d], a, b)[0] for a, b in LB]
                        ex01 = ratio(Hd[d], bd[d], 0, 30)[1]
                        rows.append((dn[d], r, ex01 / N * 1000))
                    if k == 1:
                        for nm, r, e in rows:
                            print(f'      {nm:>9s}' + fmt(r, '{:7.3f}') + f'{e:9.2f}')
                    else:
                        tot = sum(e for _, _, e in rows)
                        near = rows[0][2] + rows[1][2] + rows[2][2]
                        print(f'      r{k}: total excess {tot:6.2f}/1000 spikes; < 60 um {near:6.2f}; ratio .25-.5 ms at < 20 um {rows[0][1][1]:.3f}, 20-40 {rows[1][1][1]:.3f}')
                    if k == 1:
                        print(f'      total excess 0-1 ms: {sum(e for _, _, e in rows):.2f} per 1000 spikes')

        print('\n(c2) cross-unit, pairs < 60 um, by template similarity (raw r1 and clean r1): obs/exp at |lag| bands; excess/1000 spikes')
        sn = [f'{a:.1f}-{b:.1f}' for a, b in zip(SIM_EDGES[:-1], SIM_EDGES[1:])]
        for g in SORTERS:
            for v in ('pre', 'raw', 'clean'):
                z = load(ex, g, 1, v)
                if z is None:
                    continue
                H, base, ratio = cross_stats(z['Hx'], 0)
                Hs, bs = H[:3].sum(0), base[:3].sum(0)
                N = z['n'].sum()
                print(f'  {g} {v}: ' + ' | '.join(f'sim {sn[s]}: ' + fmt([ratio(Hs[s], bs[s], a, b)[0] for a, b in LB], '{:6.2f}')
                                                + f' exc {ratio(Hs[s], bs[s], 0, 30)[1] / N * 1000:5.2f}' for s in range(NS)))
