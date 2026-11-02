"""T1 (shape mismatch vs lag), T2 (log-amplitude structure function), T6 (early/late d' vs lag, and
complementary high-similarity pairs) on one default sort. CPU only.
usage: longscale.py RUN_DIR [FEATS.npz] --label L --out OUT.json
Features: RUN_DIR/pc_features.npy (KS4's tF = matching residual + own template, on the unit's channels),
or FEATS.npz from recon_feats.py (PCs of the preprocessed data; overlapping spikes not removed).

All lag curves are variograms over 1-min bins, corrected for estimation noise with a same-bin split-half
baseline, and fitted as nugget + growth: y(l) = n0 + c * l**p. The nugget n0 (lag-independent bin-to-bin
fluctuation) is already present in a 20-min sort; only the growth term is extrapolated."""
import os, json, argparse, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('run'); ap.add_argument('feats', nargs='?')
ap.add_argument('--label', default=''); ap.add_argument('--out'); ap.add_argument('--th', type=float, default=8.0)
a = ap.parse_args()
rng = np.random.default_rng(0)
FS, NB, MINB, MAXL = 30000., 20, 40, 15          # 1-min bins, >= 40 spikes per bin, lags 1..15 min
TH = a.th
t00 = time.time()
if a.feats:
    z = np.load(a.feats); F = z['F'].reshape(len(z['F']), -1); t = z['t']/FS; c = z['c'].astype(np.int64); src = 'recon'
    T = float(z['dur'])
else:
    F = np.load(os.path.join(a.run, 'pc_features.npy')).reshape(-1, 60); t = np.load(os.path.join(a.run, 'spike_times.npy'))/FS
    c = np.load(os.path.join(a.run, 'spike_clusters.npy')).astype(np.int64); src = 'pc_features'
    T = t.max()
b = np.minimum((t/T*NB).astype(int), NB-1)
order = np.argsort(c, kind='stable'); bd = np.r_[0, np.cumsum(np.bincount(c))]
lags = np.arange(1, MAXL+1)


def cosd(X, Y):
    return 1 - (X*Y).sum(-1)/np.linalg.norm(X, axis=-1)/np.linalg.norm(Y, axis=-1)


def dprime_curve(X, groups):
    """Cross-validated d' between bins k and k+l: direction from one half, d' on the other.
    groups[h][k] = row indices of half h in bin k."""
    M = [np.stack([X[g].mean(0) for g in groups[h]]) for h in (0, 1)]
    out = np.zeros(MAXL)
    for li, l in enumerate(lags):
        dd = []
        for kk in range(0, NB-l, max(1, (NB-l)//6)):
            for h in (0, 1):
                w = M[h][kk+l] - M[h][kk]; w = w/np.linalg.norm(w)
                p0 = X[groups[1-h][kk]] @ w; p1 = X[groups[1-h][kk+l]] @ w
                dd.append((p1.mean()-p0.mean())/np.sqrt(0.5*(p0.var()+p1.var())))
        out[li] = np.mean(dd)
    return out


U = []
for u in range(len(bd)-1):
    i = order[bd[u]:bd[u+1]]
    if len(i) < 2000: continue
    bb = b[i]; cnt = np.bincount(bb, minlength=NB)
    if cnt.min() < MINB: continue
    X = F[i].astype(np.float64)
    uhat = X.mean(0); uhat /= np.linalg.norm(uhat)
    proxy = X @ uhat                                     # projection on the unit-norm unit mean: th_amp proxy
    half = rng.random(len(i)) < 0.5
    grp = [[np.nonzero((bb == k) & (half == (h == 0)))[0] for k in range(NB)] for h in (0, 1)]
    MA = np.stack([X[g].mean(0) for g in grp[0]]); MB = np.stack([X[g].mean(0) for g in grp[1]])
    base = cosd(MA, MB)                                  # same-bin split-half 1-cos = estimation noise floor
    aA, aB = np.log(np.clip(MA @ uhat, 1e-6, None)), np.log(np.clip(MB @ uhat, 1e-6, None)); abase = (aA - aB)**2
    E = np.zeros(MAXL); V = np.zeros(MAXL)
    for li, l in enumerate(lags):
        k = np.arange(NB-l)
        E[li] = np.mean(0.5*(cosd(MA[k], MB[k+l]) + cosd(MB[k], MA[k+l])) - 0.5*(base[k] + base[k+l]))
        V[li] = np.mean(0.5*((aA[k+l]-aB[k])**2 + (aB[k+l]-aA[k])**2) - 0.5*(abase[k] + abase[k+l]))
    D = dprime_curve(X, grp)
    bsh = rng.permutation(bb)                            # null: bin labels shuffled across spikes
    Dn = dprime_curve(X, [[np.nonzero((bsh == k) & (half == (h == 0)))[0] for k in range(NB)] for h in (0, 1)])
    sl = np.polyfit(np.arange(NB)+0.5, 0.5*(aA+aB), 1, cov=True)
    U.append(dict(u=u, n=len(i), E=E, V=V, D=D, Dn=Dn, base=base.mean(), abase=abase.mean(), proxy=proxy,
                  slope=sl[0][0], slope_var=sl[1][0, 0]))
n_u = len(U); print(f'[{a.label}] source {src}, units analysed {n_u}, duration {T/60:.1f} min, {time.time()-t00:.0f} s', flush=True)
Es = np.array([r['E'] for r in U]); Vs = np.array([r['V'] for r in U]); Ds = np.array([r['D'] for r in U]); Dns = np.array([r['Dn'] for r in U])
D2 = Ds**2 - (Dns**2).mean()                             # d'^2 minus the estimator's null level
w_n = np.array([r['n'] for r in U]); allp = np.concatenate([r['proxy'] for r in U])
res = {'label': a.label, 'source': src, 'units': n_u, 'Th': TH}


def nug_fit(Y, p):
    """Per-row least squares Y = n0 + c*l**p; returns n0, c (c not clipped)."""
    A = np.c_[np.ones(MAXL), lags**p.__float__()]
    coef, *_ = np.linalg.lstsq(A, Y.T, rcond=None); return coef[0], coef[1]


def best_p(y, grid=np.arange(0.2, 3.01, 0.05)):
    sse = []
    for p in grid:
        A = np.c_[np.ones(MAXL), lags**p]; coef, r, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse.append(((A @ coef - y)**2).sum() if coef[1] > 0 else np.inf)
    return grid[int(np.argmin(sse))]


def boot_stats(Y, B=400):
    ps, g = [], []
    for _ in range(B):
        m = Y[rng.integers(0, len(Y), len(Y))].mean(0); ps.append(best_p(m)); g.append(m[-1] - m[0])
    return np.percentile(ps, [2.5, 50, 97.5]), np.percentile(g, [2.5, 50, 97.5])


def curve_block(name, Y, floor, note):
    m = Y.mean(0); pci, gci = boot_stats(Y)
    n0, cc = nug_fit(m[None], np.float64(1.0))
    out = {'noise_floor_median': float(floor), 'mean_by_lag': m.tolist(), 'p50_by_lag': np.percentile(Y, 50, 0).tolist(),
           'p90_by_lag': np.percentile(Y, 90, 0).tolist(), 'growth_l15_minus_l1_CI': gci.tolist(), 'exponent_with_nugget_CI': pci.tolist(),
           'nugget_linear_fit': float(n0[0])}
    print(f' {name}: noise floor (same-bin split-half, median) {floor:.5f}  [{note}]')
    for l in (1, 3, 6, 12, 15):
        print('   lag %2d min: mean %.5f p50 %.5f p90 %.5f' % (l, m[l-1], np.percentile(Y[:, l-1], 50), np.percentile(Y[:, l-1], 90)))
    print('   growth y(15)-y(1) 95%% CI %s; exponent p of n0 + c*l^p (bootstrap 95%% CI) %s' % (np.round(gci, 5), np.round(pci, 2)))
    return out


def loss_from_factor(p, f):          # fraction of spikes now >= TH that fall below TH when scaled by f
    ok = p >= TH; return ((p < TH/max(f, 1e-9)) & ok).sum()/max(ok.sum(), 1)


res['proxy'] = {'frac_below_Th': float((allp < TH).mean()), 'p1': float(np.percentile(allp, 1)), 'p5': float(np.percentile(allp, 5)),
                'frac_lost_uniform_10pct': float(((allp >= TH) & (allp < TH/0.9)).sum()/(allp >= TH).sum())}
print(' proxy: frac<Th %.3f, p1 %.2f, p5 %.2f; uniform 10%% projection loss -> %.3f of spikes >= Th' % tuple(res['proxy'].values()))

# ---- T1: E = excess 1-cos of bin-mean waveforms (~ squared angular distance). RW: p=1, linear trend: p=2.
res['T1'] = curve_block('T1 1-cos', Es, np.median([r['base'] for r in U]), 'E ~ squared distance: RW p=1, trend p=2')
ages = np.array([30, 60, 90, 120, 150]); res['T1']['loss'] = {}
for p in (1.0, 2.0):
    n0, cu = nug_fit(Es, np.float64(p)); cu = np.clip(cu, 0, None)
    L = np.array([[loss_from_factor(r['proxy'], 1 - min(cu[j]*ag**p, 0.99)) for ag in ages] for j, r in enumerate(U)])
    cm = max(nug_fit(Es.mean(0)[None], np.float64(p))[1][0], 0)            # population-mean growth applied to every unit
    Lm = np.array([[loss_from_factor(r['proxy'], 1 - min(cm*ag**p, 0.99)) for ag in ages] for r in U])
    tot = (L*w_n[:, None]).sum(0)/w_n.sum(); totm = (Lm*w_n[:, None]).sum(0)/w_n.sum()
    res['T1']['loss'][f'p={p:.0f}'] = {'ages_min': ages.tolist(), 'spike_weighted_per_unit_fit': tot.tolist(), 'spike_weighted_mean_growth': totm.tolist(),
                                      'mean_growth_1mcos_at_ages': (cm*ages**p).tolist(), 'unit_p90': np.percentile(L, 90, 0).tolist(),
                                      'units_gt50pct': (L > 0.5).mean(0).tolist()}
    print('   p=%d ages %s min: 1-cos(mean growth) %s | loss, mean growth %s | loss, per-unit fits %s | unit p90 %s | units >50%% lost %s' % (
          p, ages.tolist(), np.round(cm*ages**p, 4), np.round(totm, 3), np.round(tot, 3), np.round(np.percentile(L, 90, 0), 3), np.round((L > 0.5).mean(0), 3)))

# ---- T2: V = excess variance of log amplitude change. RW: p=1, trend: p=2.
res['T2'] = curve_block('T2 var(dlog amp)', Vs, np.median([r['abase'] for r in U]), 'RW p=1, trend p=2')
_, s2 = nug_fit(Vs, np.float64(1.0)); s2 = np.clip(s2, 0, None)          # RW diffusion per min, nugget removed
sl = np.array([r['slope'] for r in U]); se2 = np.array([r['slope_var'] for r in U])
shrink = max(0.0, 1 - se2.mean()/sl.var()); sls = sl.mean() + shrink*(sl - sl.mean())
res['T2']['slope_per_20min_logamp_pct'] = (np.percentile(sl*20, [5, 25, 50, 75, 95])).tolist(); res['T2']['shrink'] = float(shrink)
print('   log-amp slope per 20 min pct 5/25/50/75/95: %s; EB shrink %.2f; RW sd of dlog amp per sqrt(h): p50 %.3f p90 %.3f' % (
      np.round(res['T2']['slope_per_20min_logamp_pct'], 3), shrink, np.sqrt(np.median(s2)*60), np.sqrt(np.percentile(s2, 90)*60)))
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(40); gh_w = gh_w/gh_w.sum()
res['T2']['pred'] = {}
for H in (60, 300):
    Lrw = np.array([sum(wq*loss_from_factor(r['proxy'], np.exp(min(xq*np.sqrt(s2[j]*H), 0))) for xq, wq in zip(gh_x, gh_w)) for j, r in enumerate(U)])
    Ltr = np.array([loss_from_factor(r['proxy'], np.exp(min(sls[j]*H, 0))) for j, r in enumerate(U)])
    for name, L in (('random_walk', Lrw), ('trend', Ltr)):
        tot = float((L*w_n).sum()/w_n.sum())
        res['T2']['pred'][f'{name}_{H}min'] = {'spike_weighted': tot, 'unit_p50': float(np.median(L)), 'unit_p90': float(np.percentile(L, 90)),
                                              'units_gt50pct': float((L > 0.5).mean())}
        print('   %-11s %3d min: spike-weighted loss %.3f, unit p50 %.3f p90 %.3f, units >50%% lost %.3f' % (name, H, tot, np.median(L), np.percentile(L, 90), (L > 0.5).mean()))

# ---- T6: d' between bins (cross-validated); fitted on d'^2 = n0 + c*l^(2*gamma). RW gamma=0.5, trend gamma=1.
nullm, null90 = Dns.mean(), np.percentile(Dns, 90)
res['T6'] = curve_block("T6 d'^2", D2, (Dns**2).mean(), "d'^2: RW p=1, trend p=2; d' null shuffled mean %.3f p90 %.3f" % (nullm, null90))
res['T6']['d_mean_by_lag'] = Ds.mean(0).tolist(); res['T6']['null_mean'] = float(nullm); res['T6']['null_p90'] = float(null90)
print("   d' mean at lag 1/15: %.3f / %.3f; p90 %.3f / %.3f" % (Ds[:, 0].mean(), Ds[:, -1].mean(), np.percentile(Ds[:, 0], 90), np.percentile(Ds[:, -1], 90)))
res['T6']['extrap'] = {}
for p in (1.0, 2.0):
    n0, cu = nug_fit(D2, np.float64(p))
    L2 = np.where(cu > 0, (np.clip(4 - n0, 0, None)/np.clip(cu, 1e-12, None))**(1/p), np.inf)
    cm_n0, cm = nug_fit(D2.mean(0)[None], np.float64(p)); Lm = ((4 - cm_n0[0])/cm[0])**(1/p) if cm[0] > 0 else np.inf
    r6 = {'median_lag_to_d2_min': float(np.median(L2)), 'frac_d2_by_60': float((L2 <= 60).mean()), 'frac_d2_by_300': float((L2 <= 300).mean()),
          'mean_curve_lag_to_d2_min': float(Lm)}
    res['T6']['extrap'][f'p={p:.0f}'] = r6
    print("   p=%d (gamma %.1f): per-unit median lag to d'=2 %.0f min; units with d'>2 within 1 h %.3f, within 5 h %.3f; mean curve reaches 2 at %.0f min" % (
          p, p/2, *r6.values()))

# ---- T6b: complementary high-similarity pairs
cl = np.load(os.path.join(a.run, 'spike_clusters.npy')).astype(np.int64); tt = np.load(os.path.join(a.run, 'spike_times.npy'))/FS
pos = np.load(os.path.join(a.run, 'spike_positions.npy')); sim = np.load(os.path.join(a.run, 'similar_templates.npy'))
nu = cl.max()+1; K = 10; bk = np.minimum((tt/tt.max()*K).astype(int), K-1)
R = np.zeros((nu, K)); np.add.at(R, (cl, bk), 1)
oc = np.argsort(cl, kind='stable'); bc = np.r_[0, np.cumsum(np.bincount(cl, minlength=nu))]
ys = np.array([np.median(pos[oc[bc[u]:bc[u+1]], 1]) if bc[u+1] > bc[u] else np.nan for u in range(nu)])
ntot = R.sum(1); iu, ju = np.triu_indices(nu, 1)
ok = (ntot[iu] >= 200) & (ntot[ju] >= 200) & (np.abs(ys[iu]-ys[ju]) < 60)
iu, ju = iu[ok], ju[ok]; S = sim[iu, ju]
f = R[iu]/np.maximum(R[iu]+R[ju], 1); f0, f1 = f[:, :2].mean(1), f[:, -2:].mean(1)
comp = ((f0 >= 0.75) & (f1 <= 0.25)) | ((f0 <= 0.25) & (f1 >= 0.75))
Rz = (R - R.mean(1, keepdims=True))/np.maximum(R.std(1, keepdims=True), 1e-9)
rr = (Rz[iu]*Rz[ju]).mean(1)
fold = lambda u: np.maximum(R[u, :2].sum(1), 1)/np.maximum(R[u, -2:].sum(1), 1)
anti = (rr < -0.7) & ((fold(iu) > 2) | (fold(iu) < 0.5)) & ((fold(ju) > 2) | (fold(ju) < 0.5))
res['T6b'] = {}
for name, m in (('sim>0.9', S > 0.9), ('0.7<sim<=0.9', (S > 0.7) & (S <= 0.9)), ('sim<0.5 (control)', S < 0.5)):
    res['T6b'][name] = {'pairs': int(m.sum()), 'crossfade': int((comp & m).sum()), 'anticorr_2fold': int((anti & m).sum()),
                        'crossfade_per_1000': float(1000*(comp & m).sum()/max(m.sum(), 1)), 'anticorr_per_1000': float(1000*(anti & m).sum()/max(m.sum(), 1)),
                        'examples': [[int(x), int(y), round(float(s), 3)] for x, y, s in zip(iu[(comp | anti) & m][:5], ju[(comp | anti) & m][:5], S[(comp | anti) & m][:5])]}
    print('  T6b %-18s pairs(<60um, >=200 sp) %6d  cross-fade %3d (%.2f/1000)  anticorr<-0.7 & 2-fold %3d (%.2f/1000)' % (
          name, m.sum(), (comp & m).sum(), res['T6b'][name]['crossfade_per_1000'], (anti & m).sum(), res['T6b'][name]['anticorr_per_1000']))
print(f' total {time.time()-t00:.0f} s')
if a.out:
    json.dump(res, open(a.out, 'w'), indent=1)
