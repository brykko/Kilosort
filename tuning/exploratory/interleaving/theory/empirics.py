"""Numbers from existing sorts for the theory notes. Read-only.
A gain ratios per measure; B spikes per unit and virtual rounds per pass; C the blind band on real templates;
D the KS2.5 learning curves (batches to settle, 20 min against 2 hours)."""
import json, os, sys
import numpy as np

SORTS = 'C:/data/ks4_tuning/exploratory/iteration/sorts/'
SCRATCH = sys.argv[1] if len(sys.argv) > 1 else None

# ---- A. gains from round3_results.md (round 3 - round 2, round 4 - round 3)
A = {'kobe S': ([28.3, 33.5, 27.2], [13.4, .2, 1.2]), 'kobe S_track': ([27.0, 32.6, 24.6], [11.2, 1.6, 2.0]),
     'MEC S': ([85.9, 33.4, 51.5], [-4.7, 55.4, 13.8])}
print('A. kappa = mean(round 4 gain)/mean(round 3 gain)')
for k, (g3, g4) in A.items():
    g3, g4 = np.array(g3), np.array(g4)
    r = g4.mean() / g3.mean()
    # delta-method SE, seeds paired
    c = np.cov(g3, g4)
    v = (c[1, 1] + r**2 * c[0, 0] - 2 * r * c[0, 1]) / 3 / g3.mean()**2
    print(f'  {k:13s} ratio {r:5.2f} se {v**.5:4.2f}  per-seed {np.round(g4/g3,2)}  '
          f'geometric asymptote above round 2 {g3.mean()/(1-r):6.1f}, round 5 expected {g4.mean()*r:5.1f}')
for k, a, b, c in [('kobe correct at 2%', 4.27, 5.00, 5.09), ('kobe correct at 0.42M', 6.58, 7.095, 7.27),
                   ('MEC correct at 2%', 2.52, 2.975, 3.14), ('kobe whole correct', 8.28, 8.605, 8.72),
                   ('kobe whole misassigned', 3.305, 3.105, 3.045),
                   ('kobe reproducibility', .581, .619, .642), ('MEC reproducibility', .487, .520, .533),
                   ('kobe dictionary size', 2456, 1762, 1725), ('MEC dictionary size', 2469, 2005, 1952),
                   ('kobe orphans %', 13, 10.25, 9.3), ('kobe spikes M', 11.82, 12.00, 12.09)]:
    print(f'  {k:24s} {(c-b)/(b-a):5.2f}')
print('  unit change kobe (1-matched): 2->3 0.39, 3->4 0.26, ratio 0.67; MEC 0.49, 0.37, ratio 0.76')


# ---- B. spikes per unit
def counts(d):
    c = np.load(d + '/spike_clusters.npy').ravel()
    return np.bincount(c)


def V_spikes(N, m0=5, mmax=400):
    """virtual rounds, standard preset: m = clip(0.1 n, m0, mmax)"""
    N = np.asarray(N, float)
    n1 = 10 * m0
    n2 = 10 * mmax
    return np.minimum(N, n1) / m0 + 10 * np.log(np.clip(N, n1, n2) / n1) + np.maximum(N - n2, 0) / mmax


def V_batches(N, m0=20, m1=400):
    return N * np.log(m1 / m0) / (m1 - m0)


print('\nB. spikes per unit, kobe H8 s1 round 3 (20 min, 549 batches)')
n = counts(SORTS + 'kobe_repeat_s1/round3')
n = n[n > 0]
print('  units', n.size, 'total', n.sum())
p = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print('  percentiles', dict(zip(p, np.percentile(n, p).astype(int))))
print('  per batch  ', dict(zip(p, np.round(np.percentile(n, p) / 549, 2))))
print('  log10 SD of N', round(np.log10(n).std(), 3), 'mean rate', round(n.mean() / 1200, 2),
      'median rate', round(np.median(n) / 1200, 2))
for ns in [100, 300, 1000]:
    frac = ns / n
    o = np.argsort(frac)
    wmed = frac[o][np.searchsorted(np.cumsum(n[o]) / n.sum(), .5)]
    print(f'  units with N >= {ns}: {100*(n>=ns).mean():.1f}%, their spike share {100*n[n>=ns].sum()/n.sum():.1f}%; '
          f'pass fraction to reach {ns}: median unit {np.median(frac):.3f}, 90th pct {np.percentile(frac,90):.2f}, '
          f'spike-weighted median {wmed:.4f}')
print('  batches to 300 spikes (unit percentile 90/50/10 by N):', np.round(300 / np.percentile(n, [90, 50, 10]) * 549).astype(int))
for lab, f in [('standard (spikes, m 5-400)', V_spikes), ('original (batches, m 20-400)', V_batches)]:
    v = f(n)
    print(f'  virtual rounds per pass, {lab}: p10 {np.percentile(v,10):.1f} median {np.median(v):.1f} p90 {np.percentile(v,90):.1f}')
print('  n_eff at end of a pass = 2 m, standard: p10 %.0f median %.0f p90 %.0f (rounds: N itself)' %
      tuple(2 * np.clip(.1 * np.percentile(n, [10, 50, 90]), 5, 400)))


# ---- C. blind band: if unit u had no template, what would the rest of the dictionary do to its spikes?
def best_rho(T):
    K, nt, nc = T.shape
    norm = np.sqrt((T**2).sum((1, 2)))
    pk = (T**2).sum(1).argmax(1)
    F = np.fft.rfft(T, n=2 * nt, axis=1)
    rho = np.zeros(K)
    who = np.zeros(K, int)
    for u in range(K):
        c = np.where(np.abs(pk - pk[u]) <= 24)[0]
        c = c[c != u]
        if c.size == 0:
            continue
        X = np.fft.irfft((F[c] * np.conj(F[u])[None]).sum(2), axis=1)
        r = X.max(1) / (norm[u] * norm[c] + 1e-9)
        j = r.argmax()
        rho[u] = r[j]
        who[u] = c[j]
    return norm, rho, who


def classify(s, rho, Th):
    below = s < Th
    absorbed = (rho * s >= Th) & ~below
    res = s * np.sqrt(1 - np.clip(rho, 0, 1)**2)
    blind = absorbed & (res < Th)
    seen = absorbed & (res >= Th)
    whole = ~absorbed & ~below
    return below, whole, seen, blind


print('\nC. blind band on real templates (lag-maximised best other template; units with >= 300 spikes)')
for lab, d, Th in [('kobe H8 r3', SORTS + 'kobe_repeat_s1/round3', 7), ('kobe H8 r2', SORTS + 'kobe_repeat_s1/round2', 7),
                   ('mec H8 r3', SORTS + 'mec_repeat_s1/round3', 7), ('kobe default r2', SORTS + 'kobe_default_repeat_s1/round2', 8)]:
    if not os.path.exists(d + '/templates.npy'):
        print('  missing', d)
        continue
    T = np.load(d + '/templates.npy')
    nn = np.bincount(np.load(d + '/spike_clusters.npy').ravel(), minlength=len(T))[:len(T)]
    keep = nn >= 300
    s, rho, who = best_rho(T)
    Wn = np.sqrt((np.load(d + '/Wall.npy')**2).sum((1, 2)))
    if len(Wn) != len(T):
        print('  Wall and templates differ in length', len(Wn), len(T))
        continue
    print(f'  {lab}: {keep.sum()} units; template norm median {np.median(Wn[keep]):.1f}; best other rho median {np.median(rho[keep]):.3f}, '
          f'p10-p90 {np.percentile(rho[keep],10):.2f}-{np.percentile(rho[keep],90):.2f}; '
          f'rho>0.9: {100*(rho[keep]>.9).mean():.1f}%, >0.975: {100*(rho[keep]>.975).mean():.1f}%')
    for th in sorted({Th, 10}):
        b, w, se, bl = classify(Wn[keep], rho[keep], th)
        print(f'    Th {th:2d}: below Th {100*b.mean():5.1f}%  left whole in residual {100*w.mean():5.1f}%  '
              f'absorbed, residual still crosses {100*se.mean():5.1f}%  absorbed and BLIND {100*bl.mean():5.1f}%')

# ---- D. KS2.5 learning curves (parsed log in the scratchpad)
if SCRATCH:
    r = json.load(open(SCRATCH + '/learn.json'))
    print('\nD. KS2.5 learning (auto-stop): batches at stop, passes, wall time')
    rows = [x for x in r if 'long_data/windows' in x['label'] or '/runs/ks25_standard' in x['label']]

    def settle(y, ib, frac=.9):
        y = np.array([np.median(y[max(0, i - 5):i + 1]) for i in range(len(y))])
        f, i0 = y[-1], y[0]
        idx = np.where((y - i0) / (f - i0) >= frac)[0]
        return ib[idx[0]] if len(idx) else np.nan

    for x in rows:
        rec = np.array(x['rec'])
        ib, units, nst0 = rec[:, 1], rec[:, 3], rec[:, 7]
        print(f"  {x['label'].split('/')[-2]:28s} stop {x['stop']:5d} ({x['passes']:.2f} passes)  90% of final units by batch {settle(units, ib):5.0f}, "
              f"90% of final spikes/batch by {settle(nst0, ib):5.0f}; learn {x['tlearn']:.0f} s = {x['tlearn']/x['stop']:.2f} s/batch; nstot q2 {rec[-1,17]:.0f}")
