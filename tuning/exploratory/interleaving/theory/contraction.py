"""Closed-form hard-EM rates for a pair of equal-variance Gaussians, and a read of the round evidence.

Two units at +/-a along their difference, unit noise, so d' = 2a (a is half the separation in noise SDs).
Lloyd map on the boundary b and on the normal direction, at the symmetric fixed point:
  mu_hat(a) = E[x | x>0] = a(2Phi(a)-1) + 2 phi(a)        centroid (biased outward)
  kappa_b(a)   = 2 phi(a) mu_hat                           boundary-shift mode
  kappa_rot(a) = 2 phi(a) / mu_hat                         rotation of the normal (the confirmation mode)
Each is checked by Monte Carlo below.
Also: the fixed point for unequal mixing (a rare unit beside a common one), by iterating the 1-D Lloyd map.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def rates(a):
    phi, Phi = norm.pdf(a), norm.cdf(a)
    mu = a * (2 * Phi - 1) + 2 * phi
    return mu, 2 * phi * mu, 2 * phi / mu


def mc_kappa(a, n=40_000_000, eps=0.02, seed=0):
    r = np.random.default_rng(seed)
    n2 = n // 2
    x1 = np.concatenate((a + r.standard_normal(n2), -a + r.standard_normal(n2)))
    x2 = r.standard_normal(n)
    def step(b, th):
        s = x1 + th * x2 - b > 0
        c1p, c1m = x1[s].mean(), x1[~s].mean()
        c2p, c2m = x2[s].mean(), x2[~s].mean()
        return (c1p + c1m) / 2, (c2p - c2m) / (c1p - c1m)
    b0, _ = step(0, 0)
    bp, _ = step(eps, 0); bm, _ = step(-eps, 0)
    _, tp = step(0, eps); _, tm = step(0, -eps)
    return (bp - bm) / (2 * eps), (tp - tm) / (2 * eps)


print('a=d\'/2  d\'    bias   Bayes_err  kappa_b  (MC)    kappa_rot (MC)   1/(1-k_rot)')
for a in [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]:
    mu, kb, kr = rates(a)
    mb, mr = mc_kappa(a)
    print(f'{a:5.2f} {2*a:5.1f} {mu-a:6.3f} {norm.cdf(-a):8.4f} {kb:8.3f} {mb:7.3f} {kr:8.3f} {mr:7.3f} {1/(1-kr):8.2f}')

print('\nd\' at which the boundary mode has rate kappa (reading the gain ratio as a toy contraction rate):')
for k in [0.10, 0.165, 0.25, 0.33, 0.38, 0.5]:
    a = brentq(lambda x: rates(x)[1] - k, 0.8, 4)
    print(f'  kappa_b {k:.3f}: d\' {2*a:.2f}, Bayes error {norm.cdf(-a)*100:.1f}%')

print('\nUnequal mixing: 1-D Lloyd fixed point, rare unit at -a (fraction pi), common unit at +a')
print(" d'   pi     boundary   rare-side purity  common-side purity  recall rare   Bayes boundary   rate")
def cell(b, a, pi):
    pl, pr = pi, 1 - pi
    Lm = pl * norm.cdf(b + a); Lp = pr * norm.cdf(b - a)              # mass below b from each
    Um = pl * norm.sf(b + a);  Up = pr * norm.sf(b - a)               # mass above b
    def tm(lo, hi, m):                                                # int x phi(x-m) on (lo,hi)
        al, bh = lo - m, hi - m
        return m * (norm.cdf(bh) - norm.cdf(al)) + norm.pdf(al) - norm.pdf(bh)
    cL = (pl * tm(-np.inf, b, -a) + pr * tm(-np.inf, b, a)) / (Lm + Lp)
    cR = (pl * tm(b, np.inf, -a) + pr * tm(b, np.inf, a)) / (Um + Up)
    return (cL + cR) / 2, Lm / (Lm + Lp), Up / (Um + Up), norm.cdf(b + a)
for dp in [3, 4, 6]:
    a = dp / 2
    for pi in [0.5, 0.2, 0.05, 0.01]:
        b, hist = 0.0, []
        for _ in range(5000):
            nb = cell(b, a, pi)[0]
            hist.append(abs(nb - b)); b = nb
        h = np.array(hist); rate = np.median(h[1:6] / h[:5]) if h[0] > 1e-9 else float('nan')
        # local rate at the fixed point
        e = 1e-3; rate = (cell(b + e, a, pi)[0] - cell(b - e, a, pi)[0]) / (2 * e)
        _, pu_l, pu_r, rec = cell(b, a, pi)
        bay = np.log((1 - pi) / pi) / (2 * a)
        print(f'{dp:3d} {pi:5.2f} {b:9.3f} {pu_l:12.3f} {pu_r:16.3f} {rec:14.3f} {bay:14.3f} {rate:9.3f}')

print('\nTemplate-noise cost for a pair at separation D with n spikes per template, d=60 feature dims')
print("(cos^2 of estimated to true discriminant = D^2/(D^2+2d/n); excess error Phi(-D cos/2) - Phi(-D/2))")
d = 60
print('rho   s    D    n=10    n=30   n=100   n=300  n=1000   (error, %; last col = no noise)')
for s, rho in [(10, .95), (10, .9), (10, .85), (10, .8), (15, .95), (8, .9)]:
    D = s * np.sqrt(2 * (1 - rho))
    row = [100 * norm.cdf(-D * np.sqrt(D**2 / (D**2 + 2 * d / n)) / 2) for n in [10, 30, 100, 300, 1000]]
    print(f'{rho:.2f} {s:3d} {D:4.1f} ' + ' '.join(f'{x:6.1f}' for x in row) + f'  | {100*norm.cdf(-D/2):5.1f}')

print('\nBlind band: Th/rho < s < Th_b/sqrt(1-rho^2); non-empty iff rho > 1/sqrt(2) when Th=Th_b')
for Th in [7, 8, 10]:
    print(f' Th={Th}: ' + '  '.join(f'rho {r:.2f}: {Th/r:.1f}-{Th/np.sqrt(1-r*r):.1f}' for r in [.75, .8, .85, .9, .95]))

print('\nKS2 lambda-prior acceptance, Cf/s^2 = (rho^2 + 2 lam rho - lam)/(1+lam) against KS4 rho^2 (lam=10)')
for r in [1, .95, .9, .85, .8, .7, .6, .5]:
    print(f' rho {r:.2f}: KS2 {(r*r+20*r-10)/11:6.3f}  KS4 {r*r:6.3f}')

print('\nThree update families in a scalar linear model with fixed-point rate kappa (error per pass of the data):')
print('  rounds: kappa. memoized, many batches, same order each pass: mu solves mu = exp(kappa(1 - 1/mu)).')
print('  forgetful (memory m spikes, N spikes per pass): exp(-(1-kappa) N/m), noise floor var = 1/(2 m (1-kappa)).')
print(' kappa  memoized  kappa^2  passes saved  forgetful N/m=3   N/m=10')
for k in [.15, .25, .35, .5, .7, .85]:
    mu = brentq(lambda m: np.exp(k * (1 - 1 / m)) - m, 1e-6, 1 - 1e-6) if k < 1 else 1
    print(f' {k:.2f} {mu:9.3f} {k*k:8.3f} {np.log(mu)/np.log(k):10.2f} {np.exp(-(1-k)*3):14.3f} {np.exp(-(1-k)*10):9.4f}')

print('\nBBP split detection: ell = pi(1-pi) D^2, threshold m > d/ell^2, usable (cos^2 ~ 0.8) at m ~ 10 d/ell^2')
for D, lab in [(5.5, 'rho .85, s 10'), (3.2, 'rho .95, s 10'), (2.0, 'rho .98, s 10')]:
    for pi in [.5, .2, .05]:
        ell = pi * (1 - pi) * D**2
        print(f' {lab:14s} pi {pi:.2f}: ell {ell:6.2f}, m_thresh {d/ell**2:8.1f}, m_usable {10*d/ell**2:8.1f}, minority spikes at usable {pi*10*d/ell**2:7.1f}')
