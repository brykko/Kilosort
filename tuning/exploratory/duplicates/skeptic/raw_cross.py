"""Raw-waveform test of CROSS-unit coincidences flagged by the harness's §6.3 cleaning (cleaning.json):
one spike reported by two units, or two spikes?

For a sample of flagged pairs (a, b) <= 150 um apart, coincident spike pairs at the flagged lag +-2 samples
('coinc') and isolated spikes of a ('iso_a') and b ('iso_b'). Templates from raw (as raw_pairs.py), on the
union of both units' 8 highest-energy channels. Per event: single-template fits (free amplitude,
shift +-3) a_A (template A at t_a) and a_B (B at t_b), and the joint fit (A at t_a + B at t_b) when the
two templates are not collinear. One spike: a_A ~ 1 (or a_B ~ 1) and the joint fit gives one amplitude
~0; two spikes: single-template amplitudes > 1 and both joint amplitudes ~1. Energy ratio
E / E(typical isolated A) also reported.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from raw_pairs import Raw, TW, isolated, shifted, BIN_KOBE

OUTDIR = Path(__file__).resolve().parent


def fit1(y, v_by_d):
    best = None
    for d, v in v_by_d.items():
        a = v @ y / (v @ v)
        R = ((y - a * v) ** 2).sum()
        if best is None or R < best[0]:
            best = (R, a, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--sort', required=True)
    ap.add_argument('--bin', default=str(BIN_KOBE))
    ap.add_argument('--pairs', type=int, default=30)
    ap.add_argument('--per', type=int, default=12)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sort = Path(a.sort)
    t = np.load(sort / 'spike_times.npy').ravel().astype(np.int64)
    c = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    o = np.argsort(t, kind='stable'); t, c = t[o], c[o]
    T = np.load(sort / 'templates.npy')
    chm = np.load(sort / 'channel_map.npy').ravel()
    nmax = np.memmap(a.bin, np.int16, 'r').size // 385
    raw = Raw(a.bin, chm)
    iso = isolated(t, c)
    fp = [p for p in json.loads((sort / 'cleaning.json').read_text())['flagged_pairs'] if p['distance_um'] <= 150]
    pick = rng.choice(len(fp), size=min(a.pairs, len(fp)), replace=False)
    rows = []
    for k in pick:
        p = fp[k]
        A, B, L = p['a'], p['b'], p['lag_samples']
        chans = np.unique(np.r_[np.argsort(-(T[A] ** 2).sum(0))[:8], np.argsort(-(T[B] ** 2).sum(0))[:8]])
        tA, tB = t[c == A], t[c == B]
        mk = lambda tt_: tt_[(tt_ > 2000) & (tt_ < nmax - 2000)]
        iA, iB = mk(t[(c == A) & iso]), mk(t[(c == B) & iso])
        if iA.size < 30 or iB.size < 30:
            continue
        TA = np.mean([raw.snip(s + TW[0], s + TW[-1] + 1, chans) for s in rng.choice(iA, 40, replace=False)], 0)
        TB = np.mean([raw.snip(s + TW[0], s + TW[-1] + 1, chans) for s in rng.choice(iB, 40, replace=False)], 0)
        j = np.searchsorted(tB, tA + L - 2); ok = (j < tB.size)
        jj = np.minimum(j, tB.size - 1)
        co = ok & (np.abs(tB[jj] - tA - L) <= 2)
        cA, cB = tA[co], tB[jj[co]]
        sel = (cA > 2000) & (cB < nmax - 2000)
        cA, cB = cA[sel], cB[sel]
        if cA.size < 5:
            continue
        idx = rng.choice(cA.size, size=min(a.per, cA.size), replace=False)
        events = [('coinc', cA[i], cB[i]) for i in idx]
        rA, rB = np.setdiff1d(iA, cA), np.setdiff1d(iB, cB)
        events += [('iso_a', s, None) for s in rng.choice(rA, min(a.per // 2, rA.size), replace=False)]
        events += [('iso_b', None, s) for s in rng.choice(rB, min(a.per // 2, rB.size), replace=False)]
        # null for the joint fit: an isolated A spike with a FAKE B detection at the flagged lag
        events += [('fake', s, s + L) for s in rng.choice(rA, min(a.per // 2, rA.size), replace=False)]
        nA = (TA ** 2).sum()
        for cls, ta_, tb_ in events:
            t0 = min(x for x in (ta_, tb_) if x is not None)
            t1 = max(x for x in (ta_, tb_) if x is not None)
            w0, w1 = t0 - 35, t1 + 55
            x = raw.snip(w0, w1, chans); Lw = x.shape[0]; y = x.ravel()
            r = {'cls': cls, 'A': A, 'B': B, 'lag': L, 'sim': float((TA * TB).sum() / np.sqrt(nA * (TB ** 2).sum()))}
            if ta_ is not None:
                vA = {d: shifted(TA, ta_ - w0, d, Lw) for d in range(-3, 4)}
                r['a_A'] = float(fit1(y, vA)[1])
            if tb_ is not None:
                vB = {d: shifted(TB, tb_ - w0, d, Lw) for d in range(-3, 4)}
                r['a_B'] = float(fit1(y, vB)[1])
            if ta_ is not None and tb_ is not None:
                best = None
                for d1, v1 in vA.items():
                    for d2, v2 in vB.items():
                        M = np.stack([v1, v2], 1)
                        G = M.T @ M
                        if np.linalg.cond(G) > 1e4:
                            continue
                        s = np.linalg.solve(G, M.T @ y)
                        R = ((y - M @ s) ** 2).sum()
                        if best is None or R < best[0]:
                            best = (R, s[0], s[1])
                if best is not None:
                    r['joint_A'], r['joint_B'] = float(best[1]), float(best[2])
            r['E_over_EA'] = float((y ** 2).sum() / nA)
            rows.append(r)
    (OUTDIR / f'raw_cross_{a.name}.json').write_text(json.dumps(rows))
    print(a.name, 'pairs used', len({(r['A'], r['B']) for r in rows}))
    for cls in ('coinc', 'fake', 'iso_a', 'iso_b'):
        R = [r for r in rows if r['cls'] == cls]
        g = lambda k: np.array([r[k] for r in R if k in r], float)
        jA, jB = g('joint_A'), g('joint_B')
        msg = f"{cls:6s} n={len(R)} a_A={np.median(g('a_A')) if g('a_A').size else np.nan:.2f} a_B={np.median(g('a_B')) if g('a_B').size else np.nan:.2f}"
        if jA.size:
            mn = np.minimum(jA, jB)
            msg += f" joint A={np.median(jA):.2f} B={np.median(jB):.2f} P(min joint<0.3)={np.mean(mn < 0.3):.2f} P(both>0.5)={np.mean(mn > 0.5):.2f} (n joint={jA.size})"
        print(msg)
    for cl_ in ('coinc', 'fake'):
      R = [r for r in rows if r['cls'] == cl_]
      print(cl_)
      for lo, hi in ((0, 2), (3, 6), (7, 30)):
        S = [r for r in R if lo <= abs(r['lag']) <= hi and 'joint_A' in r]
        if S:
            mn = np.minimum([r['joint_A'] for r in S], [r['joint_B'] for r in S])
            print(f"  |lag| {lo}-{hi}: n={len(S)} P(both joint>0.5)={np.mean(mn > 0.5):.2f} P(min<0.3)={np.mean(mn < 0.3):.2f}"
                  f" median a_A={np.median([r['a_A'] for r in S]):.2f} sim={np.median([r['sim'] for r in S]):.2f}")


if __name__ == '__main__':
    main()
