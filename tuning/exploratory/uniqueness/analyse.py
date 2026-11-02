"""Summaries of the unique-tuning prototype: calibration, scores across rho*, audit, resolution sweeps."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from unique_tuning import MVL_MIN, RHO_GRID, R_UM, SI_MIN, scores  # noqa: E402

OUT = Path('C:/data/ks4_tuning/exploratory/uniqueness')
DBINS = [0, 25, 50, 100, 200, 300, 500]


def load(p):
    z = np.load(p)
    return ({k[4:]: z[k] for k in z.files if k.startswith('tab_')}, {k[5:]: z[k] for k in z.files if k.startswith('pair_')})


def twin_stat(tab, pairs):
    """min rho over the modalities on which either unit is strongly tuned (nan if neither is)."""
    shd, spos = tab['mvl_c'] >= MVL_MIN, tab['si_c'] >= SI_MIN
    i, j = pairs['i'], pairs['j']
    nh, npos = shd[i] | shd[j], spos[i] | spos[j]
    t = np.full(i.size, np.nan)
    t[nh & ~npos] = pairs['rho_hd'][nh & ~npos]
    t[npos & ~nh] = pairs['rho_pos'][npos & ~nh]
    both = nh & npos
    t[both] = np.minimum(pairs['rho_hd'][both], pairs['rho_pos'][both])
    return t


def calibration(ex):
    print(f'\n=== {ex}: calibration on default sorts ===')
    frag, near, far = [], [], []
    prof = {x: np.zeros(len(DBINS) - 1) for x in (0.8, 0.9, 0.95)}
    npairs = np.zeros(len(DBINS) - 1)
    files = sorted((OUT / ex).glob('default_s[123].npz'))
    for f in files:
        tab, pairs = load(f)
        fr = np.load(f.with_name(f.stem + '_fragments.npz'))
        shd, spos = tab['mvl_c'][fr['unit']] >= MVL_MIN, tab['si_c'][fr['unit']] >= SI_MIN
        ft = np.where(shd & spos, np.minimum(fr['rho_hd'], fr['rho_pos']), np.where(shd, fr['rho_hd'], fr['rho_pos']))
        frag.append(ft)
        t = twin_stat(tab, pairs)
        ok = ~np.isnan(t)
        b = np.digitize(pairs['d'], DBINS) - 1
        for k in range(len(DBINS) - 1):
            sel = ok & (b == k)
            npairs[k] += sel.sum()
            for x in prof:
                prof[x][k] += np.sum(t[sel] >= x)
        near.append(t[ok & (pairs['d'] <= R_UM)])
        far.append(t[ok & (pairs['d'] > 300)])
    frag, near, far = (np.concatenate(a) for a in (frag, near, far))
    q = [5, 10, 25, 50, 75, 90]
    print(f'twin statistic quantiles {q}:')
    for name, a in (('fragments (same cell)', frag), (f'pairs <= {R_UM:.0f} um', near), ('pairs 300-500 um', far)):
        print(f'  {name:24s} n={a.size:7d}  ' + ' '.join(f'{v:6.3f}' for v in np.percentile(a, q)))
    print('rho*:      ' + ' '.join(f'{r:6.2f}' for r in RHO_GRID[:-1]))
    print('fragments >= rho*  (split detected): ' + ' '.join(f'{np.mean(frag >= r):6.1%}' for r in RHO_GRID[:-1]))
    print('far pairs >= rho*  (base twin rate): ' + ' '.join(f'{np.mean(far >= r):6.2%}' for r in RHO_GRID[:-1]))
    print('near pairs >= rho*:                  ' + ' '.join(f'{np.mean(near >= r):6.2%}' for r in RHO_GRID[:-1]))
    print('fraction of pairs with twin statistic >= x, by distance (um):')
    print('  ' + ' '.join(f'{f"{a}-{b}":>10s}' for a, b in zip(DBINS[:-1], DBINS[1:])))
    print('  pairs ' + ' '.join(f'{int(n):10d}' for n in npairs))
    for x, v in prof.items():
        print(f'  >= {x:4.2f}' + ' '.join(f'{c / n:10.3%}' for c, n in zip(v, npairs)))


def group_scores(ex, groups, rhos=RHO_GRID):
    print(f'\n=== {ex}: scores (mean over seeds) across rho* ===')
    for var in ('count', 'saturating', 'information'):
        print(f'{var}:  rho* ' + ' '.join(f'{r:>8}' for r in rhos))
        for g in groups:
            files = sorted((OUT / ex).glob(f'{g}_s*.npz')) + sorted((OUT / ex).glob(f'{g}_r*.npz'))
            files = [f for f in files if not f.stem.endswith('fragments')]
            if not files:
                continue
            vals = [[scores(*load(f), r)[var] for r in rhos] for f in files]
            m = np.mean(vals, 0)
            print(f'  {g:8s}({len(files)})  ' + ' '.join(f'{v:8.1f}' for v in m))
    print('units (mean over seeds): eligible, strongly tuned (hd, pos, conjunctive)')
    for g in groups:
        files = [f for f in sorted((OUT / ex).glob(f'{g}_[sr]*.npz')) if not f.stem.endswith('fragments')]
        if files:
            s = [scores(*load(f), 0.9) for f in files]
            print(f"  {g:8s} {np.mean([x['n_eligible'] for x in s]):7.0f} {np.mean([x['n_strong'] for x in s]):7.0f} "
                  f"({np.mean([x['n_strong_hd'] for x in s]):.0f}, {np.mean([x['n_strong_pos'] for x in s]):.0f}, {np.mean([x['n_conj'] for x in s]):.0f})")


def audit(ex):
    print(f'\n=== {ex}: perturbation audit on default seed 1 (score change from baseline) ===')
    d = OUT / f'{ex}_perturb'
    base = load(d / 'default_s1.npz')
    rhos = [0.8, 0.9, 0.95, np.inf]
    b = {r: scores(*base, r) for r in rhos}
    print('                 eligible ' + ' | '.join(f'rho* {r}: count / sat / info' for r in rhos))
    for f in sorted(d.glob('default_s1_*.npz')):
        p = load(f)
        line = []
        for r in rhos:
            s = scores(*p, r)
            line.append(f"{s['count'] - b[r]['count']:+5d} / {s['saturating'] - b[r]['saturating']:+6.1f} / {s['information'] - b[r]['information']:+7.1f}")
        print(f"  {f.stem[11:]:14s} {scores(*p, 0.9)['n_eligible'] - b[0.9]['n_eligible']:+5d}   " + ' | '.join(line))
    print('  baseline       ' + ' | '.join(f"{b[r]['count']:5d} / {b[r]['saturating']:6.1f} / {b[r]['information']:7.1f}" for r in rhos))


if __name__ == '__main__':
    what = sys.argv[1:] or ['calibration', 'scores', 'audit']
    for ex in ('kobe', 'mec'):
        if not (OUT / ex).exists():
            continue
        if 'calibration' in what:
            calibration(ex)
        if 'scores' in what:
            groups = ['default', 'ks25', 'H8'] + (['B2', 'G1', 'W', 'G2', 'X1', 'R4', 'R4g4', 'R4g8'] if ex == 'kobe' else [])
            group_scores(ex, groups)
        if 'audit' in what and (OUT / f'{ex}_perturb').exists():
            audit(ex)
