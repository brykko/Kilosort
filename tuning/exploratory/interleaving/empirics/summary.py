"""Tables for notes/empirics.md from the outputs of the measurement scripts. Prints to stdout; sections can be picked.

    python summary.py m1 m1raw m3 m4 m5 m2
"""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

from common import OUT, ROUNDS, SEEDS

Q = (.1, .5, .9)
EX = ('kobe', 'mec')


def q(x, p=Q, f='{:.4f}'):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return ' / '.join(f.format(v) for v in np.quantile(x, p)) if x.size else 'n/a'


def m1():
    print('## M1 Wall-level')
    for setting in ('H8', 'default'):
        rounds = ROUNDS[setting]
        for ex in EX:
            A = json.loads((OUT / f'm1_assign_{ex}_{setting}.json').read_text())
            print(f'\n### {ex} {setting}: assignments (cleaned spikes)')
            for a, b in zip(rounds[:-1], rounds[1:]):
                for d in ('fwd', 'back'):
                    x = [A['assign'][f's{s}_{a}to{b}'][d]['elig'] for s in SEEDS]
                    y = [A['assign'][f's{s}_{a}to{b}'][d]['all'] for s in SEEDS]
                    print(f"r{a}->r{b} {d}: eligible units' spikes: shared raw {min(i['shared'] for i in x):.3f}-{max(i['shared'] for i in x):.3f}, "
                          f"chance {min(i['chance'] for i in x):.3f}-{max(i['chance'] for i in x):.3f}, "
                          f"not shared (corrected) {min(i['not_shared_corr'] for i in x):.3f}-{max(i['not_shared_corr'] for i in x):.3f}, "
                          f"relabelled of shared {min(i['relabelled_of_shared'] for i in x):.3f}-{max(i['relabelled_of_shared'] for i in x):.3f}; "
                          f"all units' spikes: not shared {min(i['not_shared_corr'] for i in y):.3f}-{max(i['not_shared_corr'] for i in y):.3f}, "
                          f"relabelled {min(i['relabelled_of_shared'] for i in y):.3f}-{max(i['relabelled_of_shared'] for i in y):.3f}")
                print('   same-sample share of linked', [round(A['assign'][f's{s}_{a}to{b}']['dt0'], 3) for s in SEEDS])
            print(f'\n### {ex} {setting}: template steps, pooled over seeds; quantiles .1/.5/.9')
            ms = [np.load(OUT / f'm1_{ex}_{setting}_s{s}.npz') for s in SEEDS]
            cat = lambda k: np.concatenate([m[k] for m in ms])  # noqa: E731
            for mode, lab in (('all', 'all eligible, spike-share successor'), ('mut', 'mutual >= 0.5')):
                others = [r for r in rounds if r != (3 if setting == 'H8' else 2)]
                print(f'{lab}: units {cat("unit").size}, ' + ', '.join(f'have r{r}: {np.isfinite(cat(f"{mode}_{r}_c0")).mean():.3f}' for r in others))
                for r in others:
                    print(f"  mid<->r{r}: cos0 {q(cat(f'{mode}_{r}_c0'))}; cos lag {q(cat(f'{mode}_{r}_cl'))}; |dT|/|T| {q(cat(f'{mode}_{r}_l2'))}; amp ratio {q(cat(f'{mode}_{r}_ar'))}")
                if setting == 'H8':
                    a, b = cat(f'{mode}_2_l2'), cat(f'{mode}_4_l2')
                    ok = np.isfinite(a) & np.isfinite(b)
                    p = cat(f'{mode}_persist')
                    print(f'  triples {ok.sum()}; contraction |T4-T3|/|T3-T2| {q((b / a)[ok])}; sum-ratio {np.sqrt((b[ok] ** 2).sum() / (a[ok] ** 2).sum()):.3f}; '
                          f'median-ratio {np.median(b[ok]) / np.median(a[ok]):.3f}; fraction <1: {(b[ok] < a[ok]).mean():.3f}; persistence cos {q(p)}; |T4-T2|/|T| {q(cat(f"{mode}_l2_24"))}')
                    print('   by rate tercile (r) x amp tercile (a): median contraction, |T3-T2|/|T|, |T4-T3|/|T|, n')
                    rq, aq = cat('rq'), cat('aq')
                    for i in range(3):
                        print('   ' + ' | '.join(f"r{i}a{j}: {np.median((b / a)[ok & (rq == i) & (aq == j)]):.2f} {np.median(a[ok & (rq == i) & (aq == j)]):.3f} {np.median(b[ok & (rq == i) & (aq == j)]):.3f} ({(ok & (rq == i) & (aq == j)).sum()})" for j in range(3)))
            print(f'\n### {ex} {setting}: seed-to-seed, same cell (mutual >= 0.5), pooled over pairs')
            for r in rounds:
                x = np.concatenate([np.load(OUT / f'm1_seedpairs_{ex}_{setting}_s{a}s{b}_r{r}.npy') for a, b in ((1, 2), (2, 3), (3, 1))])
                print(f'r{r}: pairs {x.shape[0]}, cos0 {q(x[:, 3])}; cos lag {q(x[:, 4])}; |dT|/|T| {q(x[:, 5])}; amp ratio {q(x[:, 6])}')
            if setting == 'H8':
                f = np.load(OUT / f'm1_seedfixed_{ex}_{setting}.npz')
                x = np.concatenate([f[k] for k in f.files])
                print(f'fixed cells (linked in both seeds and both rounds): {x.shape[0]} pairs; r2 cos lag {q(x[:, 1])} |dT|/|T| {q(x[:, 2])}; r4 cos lag {q(x[:, 5])} |dT|/|T| {q(x[:, 6])}; share with r4 closer (L2): {(x[:, 6] < x[:, 2]).mean():.3f}')


def m1raw():
    print('## M1 raw estimator (seed 1), quantiles .1/.5/.9; columns of step: step2, null2, cos, splithalf2, noise2, n')
    for setting in ('H8', 'default'):
        for ex in EX:
            z = np.load(OUT / f'm1raw_{ex}_{setting}.npz')
            print(f'\n### {ex} {setting}')
            for k in z.files:
                if k in ('unit', 'rq', 'aq'):
                    continue
                x = z[k]
                ok = np.isfinite(x[:, 0])
                st, nu, c, sh, no = (x[ok, i] for i in range(5))
                print(f'{k}: units {ok.sum()}/{x.shape[0]}; sqrt step {q(np.sqrt(np.maximum(st, 0)))}; sqrt null {q(np.sqrt(np.maximum(nu, 0)))}; step2/null2 {q(st / nu, f="{:.2f}")}; '
                      f'share step2 > 2 null2 {(st > 2 * nu).mean():.3f}; cos {q(c)}; sqrt noise {q(np.sqrt(np.maximum(no, 0)))}; sqrt split-half {q(np.sqrt(np.maximum(sh, 0)))}')
            if setting == 'H8':
                for mode in ('all', 'mut'):
                    a, b = z[f'{mode}_2'], z[f'{mode}_4']
                    ok = np.isfinite(a[:, 0]) & np.isfinite(b[:, 0])
                    ea = a[ok, 0] - a[ok, 1]         # step2 minus null2, unbiased excess
                    eb = b[ok, 0] - b[ok, 1]
                    print(f'{mode}: raw contraction |T4-T3|/|T3-T2| {q(np.sqrt(np.maximum(b[ok, 0], 0) / np.maximum(a[ok, 0], 1e-12)))}; '
                          f'excess over null: sum ratio sqrt {np.sqrt(np.maximum(eb.sum(), 0) / ea.sum()):.3f}; median sqrt excess r3-r2 {np.median(np.sqrt(np.maximum(ea, 0))):.4f}, r4-r3 {np.median(np.sqrt(np.maximum(eb, 0))):.4f}')
                    rq, aq = z['rq'][ok], z['aq'][ok]
                    print('   median sqrt step r3-r2, r4-r3, sqrt noise by rate tercile (rows) x amp (cols)')
                    for i in range(3):
                        print('   ' + ' | '.join(
                            f"{np.median(np.sqrt(np.maximum(a[ok, 0][(rq == i) & (aq == j)], 0))):.3f} {np.median(np.sqrt(np.maximum(b[ok, 0][(rq == i) & (aq == j)], 0))):.3f} {np.median(np.sqrt(np.maximum(a[ok, 4][(rq == i) & (aq == j)], 0))):.3f}"
                            for j in range(3)))


def m3():
    print('## M3 sample complexity')
    for name in ('kobe_H8_s1_r4', 'mec_H8_s1_r4', 'kobe_default_s1_r3', 'mec_default_s1_r3'):
        z = np.load(OUT / f'm3_{name}.npz')
        cur = z['cur']
        print(f'\n### {name}: {z["unit"].size} eligible units; rate quantiles .1/.5/.9 {q(z["rate"], f="{:.2f}")} Hz; n spikes {q(z["n"], f="{:.0f}")}; sig2/T2 {q(z["sig2"] / z["T2"], f="{:.1f}")}')
        print('measured cosine to held-out mean: n, units, quantiles .1/.5/.9')
        for k, n in enumerate((10, 30, 100, 300, 1000)):
            print(f'  n={n}: {np.isfinite(cur[:, k]).sum()} units, {q(cur[:, k], f="{:.3f}")}')
        print('closed form, all units: n*(c), batches n*/(2r), real-train batches median start, 90th pct start; quantiles .1/.5/.9')
        for c in (0.9, 0.95, 0.99):
            nv = z[f'bm{c}']
            print(f'  c={c}: n* {q(z[f"n{c}"], f="{:.0f}")}; batches {q(z[f"b{c}"], f="{:.1f}")}; real {q(nv, f="{:.0f}")}; real p90 {q(z[f"bp{c}"], f="{:.0f}")}; units that never reach it in the excerpt {np.mean(~np.isfinite(nv)):.3f}')
        r = z['rate']
        for lo, hi, lab in ((.4, .6, 'median-rate units (40-60th pct)'), (.05, .15, '10th-pct-rate units (5-15th pct)')):
            a, b = np.quantile(r, [lo, hi])
            m = (r >= a) & (r <= b)
            print(f'  {lab}: {m.sum()} units, rate {np.median(r[m]):.2f} Hz: ' + '; '.join(
                f'c={c}: median n* {np.median(z[f"n{c}"][m]):.0f}, batches {np.median(z[f"b{c}"][m]):.1f}, real {np.nanmedian(z[f"bm{c}"][m]):.0f}, real p90 {np.nanmedian(z[f"bp{c}"][m]):.0f}' for c in (0.95, 0.99)))
        print('  rate tercile x amp tercile: median batches to 0.95, 0.99 (units)')
        for i in range(3):
            print('  ' + ' | '.join(f"r{i}a{j}: {np.median(z['b0.95'][(z['rq'] == i) & (z['aq'] == j)]):.1f} {np.median(z['b0.99'][(z['rq'] == i) & (z['aq'] == j)]):.1f} ({((z['rq'] == i) & (z['aq'] == j)).sum()})" for j in range(3)))
        print(f'  measured vs closed form: median measured cos at n=100 {np.nanmedian(cur[:, 2]):.3f}; closed form {np.median(1 / np.sqrt(1 + (z["sig2"] / z["T2"])[np.isfinite(cur[:, 2])] / 100)):.3f}')
        print(f'  occupancy: batches with >=1 {q(z["occ1"], f="{:.3f}")}, >=3 {q(z["occ3"], f="{:.3f}")}, >=10 {q(z["occ10"], f="{:.3f}")}; Poisson >=10 {q(z["pocc10"], f="{:.3f}")}; share of batches holding half the spikes {q(z["half_b"], f="{:.3f}")}')
        print('   >=10 by rate tercile: ' + '; '.join(f'r{i}: real {np.median(z["occ10"][z["rq"] == i]):.3f} Poisson {np.median(z["pocc10"][z["rq"] == i]):.3f}' for i in range(3)))


def m4():
    print('## M4 non-stationarity; columns unit nf nl cos cos_ctrl deb deb_ctrl amp l2 l2_ctrl')
    for tag in ('kobe_H8', 'mec_H8', 'kobe_default', 'mec_default', 'kobe_2h', 'mec_2h'):
        x = np.load(OUT / f'm4_{tag}.npy')
        ok = np.isfinite(x[:, 3])
        y = x[ok]
        print(f'{tag}: units {x.shape[0]}, measurable (>=30 in each epoch) {ok.sum()}; cos {q(y[:, 3], f="{:.3f}")} vs control {q(y[:, 4], f="{:.3f}")}; debiased shape {q(y[:, 5], f="{:.3f}")} vs control {q(y[:, 6], f="{:.3f}")}; '
              f'amp last/first {q(y[:, 7], f="{:.3f}")}; |F-L|/|F| {q(y[:, 8], f="{:.3f}")} vs control {q(y[:, 9], f="{:.3f}")}; share of units |F-L| > control {(y[:, 8] > y[:, 9]).mean():.3f}; > 1.5x control {(y[:, 8] > 1.5 * y[:, 9]).mean():.3f}')
    for ex in EX:
        z = np.load(OUT / f'm4_occ_{ex}_2h.npz')
        print(f'{ex} 2 h occupancy: rate {q(z["rate"], f="{:.2f}")}; >=1 {q(z["occ1"], f="{:.3f}")}; >=10 {q(z["occ10"], f="{:.3f}")}; Poisson {q(z["pocc10"], f="{:.3f}")}; share of 10-min windows silent {q(z["silent10min"], f="{:.3f}")}')


def m5():
    print('## M5 universal pass')
    for ex in EX:
        J = json.loads((OUT / f'm5_{ex}.json').read_text())
        for k, v in J.items():
            print(ex, k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if not isinstance(b, list)})
        print(' score bins', [(b['lo'], b['n'], round((b['seen'] - b['chance']) / (1 - b['chance']), 3)) for b in J['s1_r4']['score_bins']])
        h = np.array(J['s1_r4']['dy_hist'])
        print(' dy hist peak', np.argmax(h) - 80, 'share within 20 um', h[60:101].sum() / h.sum())
        for rnd in (2, 4):
            xs = [np.load(OUT / f'm5_{ex}_s{s}_r{rnd}.npz') for s in SEEDS]
            cat = lambda k: np.concatenate([x[k] for x in xs])  # noqa: E731
            n, seen, ch = cat('n'), cat('seen'), cat('chance')
            f = np.clip((seen - ch) / np.maximum(n - ch, 1), 0, 1)
            rho = cat('resp') / cat('norm')
            nrm = cat('norm')
            print(f'\n {ex} r{rnd}: eligible units {n.size}, spike-weighted seen {((seen - ch).sum() / (n - ch).sum()):.3f}; f_seen {q(f, f="{:.3f}")}; unseen >=50%: {(f <= .5).mean():.3f} of units, {n[f <= .5].sum() / n.sum():.3f} of spikes; '
                  f'>=80%: {(f <= .2).mean():.3f}, {n[f <= .2].sum() / n.sum():.3f}; >=95% {(f <= .05).mean():.3f}; corrected seen spikes per unit {q(n * f, f="{:.0f}")}; units < 100 seen {((n * f) < 100).mean():.3f}')
            print(f'   rho {q(rho, (.01, .1, .5, .9, .99), "{:.3f}")}; norm |T| {q(nrm, f="{:.1f}")}; rho|T| {q(rho * nrm, f="{:.1f}")}; units rho|T| < Th_u=7: {(rho * nrm < 7).mean():.3f}; median amp0 / resp {q(cat("med_amp0") / cat("resp"), f="{:.2f}")}; near_th {q(cat("near_th"), f="{:.3f}")}')
            ok = np.isfinite(rho)
            print(f'   Spearman f_seen~rho {spearmanr(f, rho)[0]:.3f}, f_seen~|T| {spearmanr(f, nrm)[0]:.3f}, rho~|T| {spearmanr(rho, nrm)[0]:.3f}')
            # within |T| terciles
            t = np.quantile(nrm, [1 / 3, 2 / 3])
            nq = np.searchsorted(t, nrm)
            print(f'   |T| tercile edges {t.round(1).tolist()}: within each, rho terciles: f_seen median, own-template share, unit count')
            for i in range(3):
                m = nq == i
                rt = np.quantile(rho[m], [1 / 3, 2 / 3])
                g = np.searchsorted(rt, rho[m])
                print('    ' + ' | '.join(f'rho~{np.median(rho[m][g == j]):.2f}: f {np.median(f[m][g == j]):.3f} own {(cat("own")[m][g == j] > 0).mean():.3f} rate {np.median(cat("rate")[m][g == j]):.1f} ({(g == j).sum()})' for j in range(3)))
                print(f'     Spearman within tercile f~rho {spearmanr(f[m], rho[m])[0]:.3f}, own~rho {spearmanr(cat("own")[m] > 0, rho[m])[0]:.3f}')
            print(f'   first clustering: share of seen spikes in best cluster {q(cat("c1_share"), f="{:.3f}")}; unit share of that cluster {q(cat("c1_purity"), f="{:.3f}")}; cos(T1,T) {q(cat("c1_cos"), f="{:.3f}")}')
            print(f'   units owning a template: {(cat("own") > 0).mean():.3f}; f_seen own {np.median(f[cat("own") > 0]):.3f} vs orphan {np.median(f[cat("own") == 0]):.3f}; rho own {np.median(rho[cat("own") > 0]):.3f} vs orphan {np.median(rho[cat("own") == 0]):.3f}')


def m2():
    print('## M2 association of change in S_i with template and label change (descriptive)')
    for setting in ('H8', 'default'):
        for ex in EX:
            for s in SEEDS:
                m = np.load(OUT / f'm1_{ex}_{setting}_s{s}.npz')
                rounds = ROUNDS[setting]
                mid = 3 if setting == 'H8' else 2
                o = 2 if setting == 'H8' else 3
                x = m[f'mut_{o}_l2']
                dS = (m['S_mid'] - m[f'S_{o}_mut']) if setting == 'H8' else (m[f'S_{o}_mut'] - m['S_mid'])
                ok = np.isfinite(x) & np.isfinite(dS)
                r, a, S = m['rate'][ok], m['amp'][ok], m['S_mid'][ok]
                # rank-partial on rate and amplitude
                def res(y):
                    from scipy.stats import rankdata
                    ry = rankdata(y)
                    D = np.c_[np.ones(ry.size), rankdata(r), rankdata(a), rankdata(S)]
                    return ry - D @ np.linalg.lstsq(D, ry, rcond=None)[0]
                print(f'{ex} {setting} s{s}: pairs {ok.sum()}; dS sum {dS[ok].sum():.1f}; Spearman |dS| ~ |dT| {spearmanr(np.abs(dS[ok]), x[ok])[0]:.3f}; dS ~ |dT| {spearmanr(dS[ok], x[ok])[0]:.3f}; '
                      f'partial (rate, amp, S) dS~|dT| {np.corrcoef(res(dS[ok]), res(x[ok]))[0, 1]:.3f}; |dS|~|dT| {np.corrcoef(res(np.abs(dS[ok])), res(x[ok]))[0, 1]:.3f}')


if __name__ == '__main__':
    for k in sys.argv[1:]:
        globals()[k]()
