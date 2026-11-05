"""Does the first-pass-miss premise have evidence? H8, seed 1, both excerpts, read from saved files.

1. cos_u: for a unit's template T (61 x 374, whitened), the best normalised response of the universal bank:
   max over size s, grid position g, waveform k, lag t of |sum_i weigh[s,i,g] (T[iC[i,g]] * wTEMP[k])(t)| / |T|.
   A spike a*T gives the universal detector a*|T|*cos_u, so cos_u is how much of a spike's energy the universal
   templates can see; it is amplitude-free. It is computed the same way for KS4 units and for KS2.5 units.
2. Universal coverage of a KS4 unit: the share of its spikes with a universal detection within +-6 samples of the
   offset mode and 40 um of depth. The time-shifted coverage (same spikes moved by 500 samples) is the chance level
   and is subtracted.
3. Orphans (units owning no round-2 template) against cos_u and coverage, within amplitude tertiles, at round 2 and
   round 4.
4. KS2.5's units as an outside view, since a spike KS4 never extracts leaves no trace in KS4's files. Each eligible
   KS2.5 unit is: 'matched' (agreement >= 0.5 with an eligible KS4 unit, H8 rounds 2 and 4), 'elsewhere' (not
   matched, but most of its spikes are in the KS4 output after subtracting chance), or 'absent' (not matched and
   most spikes not in the KS4 output). Each unit's predicted KS4 score is its KS2.5 amplitude scaled by the matched
   units' ratio; predicted universal response = predicted score * cos_u. So 'absent' units divide into below
   threshold (small) and above threshold with a low cos_u (the premise).
"""
import gc

from common import *  # noqa
import torch  # noqa: E402

sys.path.insert(0, str(WT / 'tuning'))
sys.path.insert(0, str(WT / 'tuning' / 'exploratory' / 'isolation'))
sys.path.insert(0, str(WT / 'tuning' / 'exploratory' / 'iteration'))
from lost_units import KS25, Sort  # noqa: E402
from s2_pairs import pair  # noqa: E402

torch.set_num_threads(4)


def universal_cos(T, chan_y, chan_x, A_ops):
    """T: (n, 61, nchan) numpy. Returns cos_u (n,), best size, best waveform."""
    wTEMP = A_ops['wTEMP'].float()
    weigh = A_ops['weigh'].float()     # sizes x nC x grid
    iC = A_ops['iC'].long()            # nC x grid
    yg, xg = torch.as_tensor(A_ops['ycup']).float(), torch.as_tensor(A_ops['xcup']).float()
    out, bs, bk = [], [], []
    filt = wTEMP.unsqueeze(1)
    nm_w = (wTEMP**2).sum(1)
    for j in range(T.shape[0]):
        t = torch.from_numpy(T[j]).float().T       # nchan x 61
        n = (t**2).sum().sqrt()
        if n == 0:
            out.append(0.); bs.append(-1); bk.append(-1)
            continue
        pk = (t**2).sum(1).argmax().item()
        near = torch.nonzero((yg - chan_y[pk]).abs() + (xg - chan_x[pk]).abs() <= 80)[:, 0]
        B = torch.nn.functional.conv1d(t.unsqueeze(1), filt, padding=30)    # nchan x k x 61
        Bg = B[iC[:, near]]                                                 # nC x ng x k x t
        A = torch.einsum('sig, igkt -> sgkt', weigh[:, :, near], Bg).abs()
        m = A.max()
        ix = (A == m).nonzero()[0]
        out.append((m / n).item()); bs.append(ix[0].item()); bk.append(ix[2].item())
    return np.array(out), np.array(bs), np.array(bk), nm_w.numpy()


def coverage(t, y, u, tu, yu, shift, dt=6, dy=40.):
    """Share of spikes (t, y) with a universal detection at t + shift +- dt, depth within dy."""
    lo = np.searchsorted(tu, t + shift - dt, 'left')
    hi = np.searchsorted(tu, t + shift + dt, 'right')
    hit = np.zeros(t.size, bool)
    block = 1 << 20
    for s in range(0, t.size, block):
        e = min(s + block, t.size)
        cnt = hi[s:e] - lo[s:e]
        i = np.repeat(np.arange(s, e), cnt)
        j = np.repeat(lo[s:e], cnt) + np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        ok = np.abs(y[i] - yu[j]) <= dy
        hit[i[ok]] = True
    return np.bincount(u[hit], minlength=u.max() + 1), np.bincount(u, minlength=u.max() + 1)


def tab(key, mask, vals, label):
    print(f'  {label}: n {mask.sum()}, ' + ', '.join(f'{k} {np.mean(v[mask]):.3f}' for k, v in zip(key, vals)), flush=True)


def main(ex):
    run = O / 'sorts' / f'{ex}_repeat_s1'
    prov = json.loads((run / 'provenance.json').read_text())
    c = torch.load(O / 'stage_cache' / f"A_{prov['cache']['A']}" / 'detect1.pt', weights_only=False, map_location='cpu')
    st0 = c['out'][0]
    A_ops = {k: c['ops'][k] for k in ('wTEMP', 'weigh', 'iC', 'ycup', 'xcup')}
    del c
    gc.collect()
    o0 = np.argsort(st0[:, 0], kind='stable')      # universal detections are not in time order within a batch
    tu = st0[o0, 0] * FS
    yu = st0[o0, 1].astype(np.float32)
    del o0
    print(ex, 'universal detections', tu.size)
    o = ops_of(run / 'round2')
    xc, yc = tonp(o['xc']).astype(np.float32), tonp(o['yc']).astype(np.float32)
    chan_y, chan_x = torch.from_numpy(yc), torch.from_numpy(xc)
    print('  wTEMP norms', np.round((A_ops['wTEMP'].numpy()**2).sum(1)**.5, 3))
    res = {}

    for name, k in (('r2', 2), ('r4', 4)):
        d = folder(ex, 'H8', k, 1)
        sp = spikes(d)
        T = np.load(d / 'templates.npy')
        cu, bs, bk, _ = universal_cos(T, chan_y, chan_x, A_ops)
        ns = np.bincount(sp['clu'], minlength=T.shape[0])
        med_score = np.zeros(T.shape[0])
        o_ = np.argsort(sp['clu'], kind='stable')
        bounds = np.r_[0, np.cumsum(ns)]
        for u in range(T.shape[0]):
            if ns[u]:
                med_score[u] = np.median(sp['score'][o_[bounds[u]:bounds[u+1]]])
        # time offset between universal and final spikes: mode of the nearest differences on big spikes
        big = np.nonzero(sp['score'] > 15)[0][::20]
        i = np.searchsorted(tu, sp['t'][big])
        cand = [(tu[np.clip(i + m, 0, tu.size - 1)] - sp['t'][big], np.abs(yu[np.clip(i + m, 0, tu.size - 1)] - sp['pos'][big, 1]))
                for m in (-3, -2, -1, 0, 1, 2)]
        dif = np.concatenate([a[b <= 20] for a, b in cand])
        hist, edges = np.histogram(dif, np.arange(-60.5, 61.5))
        off = int(round(edges[hist.argmax()] + .5))
        hit, n = coverage(sp['t'].astype(np.float64), sp['pos'][:, 1], sp['clu'], tu, yu, off)
        hit0, _ = coverage(sp['t'].astype(np.float64), sp['pos'][:, 1], sp['clu'], tu, yu, off + 500)
        cov = (hit - hit0) / np.maximum(n, 1)
        rnds = ops_of(run / ('round2' if k == 2 else '.'))
        owner, _, _, _ = owners(sp['tpl'], sp['clu'])
        owned = np.bincount(owner[owner >= 0], minlength=T.shape[0])
        Si = S_i(d, ex)
        ids = np.array(sorted(Si))
        S = np.array([Si[i] for i in ids])
        res[name] = dict(ids=ids, S=S, cu=cu[ids], cov=cov[ids], med_score=med_score[ids], orphan=owned[ids] == 0, n=n[ids],
                         bs=bs[ids], bk=bk[ids])
        print(f'  {name}: time offset {off} samples; scored units {ids.size}; all spikes covered {hit.sum() / n.sum():.3f} '
              f'(chance {hit0.sum() / n.sum():.3f})', flush=True)
        print(f'    cos_u quantiles (5,25,50,75,95) {np.round(np.quantile(cu[ids], [.05, .25, .5, .75, .95]), 3)}; '
              f'S-weighted median {np.round(np.quantile(np.repeat(cu[ids], np.maximum(S * 10, 0).astype(int)), .5), 3)}')
        q = np.quantile(cu[ids], [.25, .5, .75])
        cq = np.digitize(cu[ids], q)
        sq = np.digitize(med_score[ids], np.quantile(med_score[ids], [1 / 3, 2 / 3]))
        for a in range(3):
            for b in range(4):
                m = (sq == a) & (cq == b)
                print(f'    score tertile {a} cos_u quartile {b}: n {m.sum():4d}, median score {np.median(med_score[ids][m]):5.1f}, '
                      f'cos_u {np.median(cu[ids][m]):.3f}, coverage {np.mean(cov[ids][m]):.3f}, orphan {np.mean(owned[ids][m] == 0):.3f}, '
                      f'spikes {np.median(n[ids][m]):6.0f}, S {S[m].sum():6.1f}', flush=True)
        r = np.corrcoef(np.argsort(np.argsort(cu[ids])), np.argsort(np.argsort(cov[ids])))[0, 1]
        r2 = np.corrcoef(np.argsort(np.argsort(med_score[ids])), np.argsort(np.argsort(cov[ids])))[0, 1]
        print(f'    Spearman coverage~cos_u {r:.3f}, coverage~score {r2:.3f}')
        pred = med_score[ids] * cu[ids]
        below = pred < o['Th_universal']
        print(f'    units whose median predicted universal response (score*cos_u) is under Th_universal {o["Th_universal"]}: {below.mean():.3f} of units, '
              f'{S[below].sum() / S.sum():.3f} of S; their orphan rate {np.mean(owned[ids][below] == 0):.3f} against {np.mean(owned[ids][~below] == 0):.3f}; '
              f'their median score {np.median(med_score[ids][below]):.1f}, cos_u {np.median(cu[ids][below]):.3f}')
        del sp, T, o_
        gc.collect()

    # KS2.5 as the outside view
    d25 = KS25[ex][0]
    t25 = np.load(d25 / 'spike_times.npy').ravel().astype(np.int64)
    c25 = np.load(d25 / 'spike_clusters.npy').ravel().astype(np.int64)
    amp25 = np.load(d25 / 'amplitudes.npy').ravel().astype(np.float32)
    pos25 = np.load(d25 / 'spike_positions.npy').astype(np.float32)
    T25 = np.load(d25 / 'templates.npy')
    cu25, _, _, _ = universal_cos(T25, chan_y, chan_x, A_ops)
    nm25 = np.sqrt((T25**2).sum((1, 2)))
    ref = Sort(d25, ex)
    el = np.array(sorted(ref.elig))
    Si25 = None
    if (d25 / 'evaluation.json').exists():
        Si25 = S_i(d25, ex)
    if Si25 is None and ex == 'mec':
        for w in json.loads((O.parent.parent / '25843_roger_2019-09-11/runs/transfer_ks25.json').read_text())['winner']:
            if Path(w['sort']).name == d25.name:
                Si25 = {int(a): float(b) for a, b in w['S_i'].items()}
    S25 = np.array([Si25.get(int(u), 0.) if Si25 else np.nan for u in el])
    nsp = np.bincount(c25, minlength=T25.shape[0])
    o_ = np.argsort(c25, kind='stable')
    bounds = np.r_[0, np.cumsum(nsp)]
    medamp = np.array([np.median(amp25[o_[bounds[u]:bounds[u+1]]]) if nsp[u] else 0 for u in range(T25.shape[0])])
    a25 = dict(t=t25, clu=c25, pos=pos25)
    for name, k in (('r2', 2), ('r4', 4)):
        d = folder(ex, 'H8', k, 1)
        sp = spikes(d)
        ia, ib = pair(a25, sp, dt=6, dy=60.)
        # chance: the same with the KS4 spikes shifted by 500 samples
        sp0 = dict(sp)
        sp0['t'] = sp['t'] + 500
        ia0, ib0 = pair(a25, sp0, dt=6, dy=60.)
        pres = np.bincount(c25[ia], minlength=T25.shape[0]) / np.maximum(nsp, 1)
        pres0 = np.bincount(c25[ia0], minlength=T25.shape[0]) / np.maximum(nsp, 1)
        M = int(sp['clu'].max()) + 1
        uu, nn = np.unique(c25[ia] * M + sp['clu'][ib], return_counts=True)
        ca, cb = uu // M, uu % M
        ks4n = np.bincount(sp['clu'], minlength=M)
        ag = nn / (nsp[ca] + ks4n[cb] - nn)
        Si4 = S_i(d, ex)
        ok = np.isin(cb, np.array(sorted(Si4)))
        best = np.zeros(T25.shape[0])
        bu = -np.ones(T25.shape[0], int)
        o2 = np.argsort(ag[ok])
        best[ca[ok][o2]] = ag[ok][o2]
        bu[ca[ok][o2]] = cb[ok][o2]
        # scale: KS4 score per unit of KS2.5 amp*norm, from matched units
        n4 = np.bincount(sp['clu'], minlength=M)
        oo = np.argsort(sp['clu'], kind='stable')
        bb = np.r_[0, np.cumsum(n4)]
        m = el[best[el] >= .5]
        ratio = np.array([np.median(sp['score'][oo[bb[bu[u]]:bb[bu[u]+1]]]) / (medamp[u] * nm25[u]) for u in m])
        scale = np.median(ratio)
        pred = medamp * nm25 * scale
        cls = np.where(best[el] >= .5, 'matched', np.where(pres[el] - pres0[el] >= .5, 'elsewhere', 'absent'))
        print(f'  KS2.5 vs {name}: eligible {el.size}; scale iqr {np.round(np.quantile(ratio, [.25, .75]) / scale, 2)}; '
              f'overall spikes present {pres[el].mean():.3f}, chance {pres0[el].mean():.3f}', flush=True)
        for g in ('matched', 'elsewhere', 'absent'):
            w = cls == g
            print(f'    {g:9s}: {w.sum():4d} units, S {np.nansum(S25[w]):7.1f}, median predicted score {np.median(pred[el][w]):5.1f}, cos_u {np.median(cu25[el][w]):.3f}, '
                  f'present-chance {np.median((pres - pres0)[el][w]):.3f}, spikes {np.median(nsp[el][w]):6.0f}', flush=True)
        ab = cls == 'absent'
        for lo, hi, l in ((0, 9, 'pred score < 9'), (9, 12, '9-12'), (12, 1e9, '>= 12')):
            w = ab & (pred[el] >= lo) & (pred[el] < hi)
            low = cu25[el] < np.quantile(cu25[el], .25)
            print(f'    absent, {l:15s}: {w.sum():4d} units, S {np.nansum(S25[w]):7.1f}; in the lowest cos_u quartile of KS2.5 units: {(w & low).sum()}, S {np.nansum(S25[w & low]):6.1f}; '
                  f'predicted universal response >= {o["Th_universal"]}: {(w & (pred[el] * cu25[el] >= o["Th_universal"])).sum()}', flush=True)
        q = np.quantile(cu25[el], [.25, .5, .75])
        cq = np.digitize(cu25[el], q)
        ps = pred[el]
        sq = np.digitize(ps, np.quantile(ps, [1 / 3, 2 / 3]))
        for a in range(3):
            line = []
            for b in range(4):
                w = (sq == a) & (cq == b)
                line.append(f'{np.mean(cls[w] == "matched"):.2f}/{np.mean(cls[w] == "absent"):.2f} ({w.sum()})')
            print(f'    predicted-score tertile {a} (median {np.median(ps[sq == a]):.1f}): matched/absent by cos_u quartile: ' + '; '.join(line), flush=True)
        if name == 'r2':
            base = cls.copy()
        else:
            for g in ('matched', 'elsewhere', 'absent'):
                w = base == g
                print(f'    r2 {g}: at r4 matched {np.sum(cls[w] == "matched")}, elsewhere {np.sum(cls[w] == "elsewhere")}, absent {np.sum(cls[w] == "absent")}')
        del sp, sp0
        gc.collect()


if __name__ == '__main__':
    for ex in sys.argv[1:] or EXS:
        main(ex)
