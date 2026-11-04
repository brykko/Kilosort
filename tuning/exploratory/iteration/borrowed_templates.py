"""Room for a third round of KS4 template matching, read from existing sorts. KS4 extracts spikes with the
templates of its first clustering, then clusters them again; a third round would extract with templates
of the final clusters. That can only help units the first clustering didn't give a template of their own.

For each final unit:
- borrowed share: the fraction of its spikes caught by extraction templates whose spikes mostly went to
  another unit (each template's owner is the unit that got most of its spikes); a unit owning no template
  is an orphan;
- fit: score / amplitude per spike. score.npy is the matching score KS4 compared with Th_learned
  (th_amps, the projection on the unit-norm template); amplitudes.npy is the norm of the spike's features
  on the template's nearest channels. So fit is the cosine between the spike and the template that caught
  it, over those channels;
- G: how far a template of its own would raise its scores, taken as the median, over all its spikes, of
  (median fit of owned spikes at the same amplitude) / fit. Not over its near-threshold spikes only:
  selecting spikes by score selects low fit at a given amplitude, which biased G above 1 even in units
  that own their templates;
- extra spikes: a rough count of the spikes that would then cross Th_learned, assuming the score density
  just below Th_learned equals the density in [Th, 1.25 Th]: density * Th * (1 - 1/G). Signed (negative
  when G < 1), so units that own their templates show how much the method attributes to scatter alone.

Units: kobe, the eligible units of evaluation.json (hybrid copies excluded) with their S_i; MEC, the
harness's eligible units (no per-unit S in these sorts). Spikes: all spikes KS4 output.

Reads the scored sorts of ..\\autocurate\\ (ks4_sort_scored.py); writes
C:\\data\\ks4_tuning\\exploratory\\iteration\\borrowed_<sort>.json and prints a summary.

    python borrowed_templates.py [sort names]
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'tuning'))

from cleaning import load_clean_units  # noqa: E402
from score import eligible  # noqa: E402

SORTS = Path('C:/data/ks4_tuning/exploratory/autocurate/sorts')
OUT = Path('C:/data/ks4_tuning/exploratory/iteration')
RAW = {'mec': Path('C:/data/ks4_tuning/25843_roger_2019-09-11/search/raw_artefacts_car.npz')}
DURATION = 1200.0
DENS = 1.25       # density just below Th_learned taken from [Th, DENS * Th]
N_AMP_BINS = 30


def units_of(sort, name):
    """Eligible units and their S_i (None for MEC)."""
    ev = sort / 'evaluation.json'
    if ev.exists():
        S_i = {int(k): v for k, v in json.loads(ev.read_text())['S_i'].items()}
        return np.array(sorted(S_i)), S_i
    by_cluster, _, _ = load_clean_units(sort, RAW[name.split('_')[0]])
    return np.array(sorted(eligible(sort, by_cluster, DURATION))), None


def analyse(name):
    sort = SORTS / name
    clu = np.load(sort / 'spike_clusters.npy').ravel().astype(np.int64)
    tpl = np.load(sort / 'spike_detection_templates.npy').ravel().astype(np.int64)
    score = np.load(sort / 'score.npy').ravel().astype(np.float64)
    amp = np.load(sort / 'amplitudes.npy').ravel().astype(np.float64)
    Wall = np.load(sort / 'Wall.npy')
    ops = np.load(sort / 'ops.npy', allow_pickle=True).item()
    Th = float(ops['Th_learned'])
    iU = np.asarray(ops['iU'].cpu() if hasattr(ops['iU'], 'cpu') else ops['iU'])
    xc, yc = np.asarray(ops['xc']), np.asarray(ops['yc'])
    nU, nT = Wall.shape[0], iU.size
    assert clu.max() < nU and tpl.max() < nT and clu.size == tpl.size == score.size == amp.size
    units, S_i = units_of(sort, name)

    # ownership
    C = np.bincount(tpl * nU + clu, minlength=nT * nU).reshape(nT, nU)
    used = C.sum(1) > 0
    owner = np.where(used, C.argmax(1), -1)
    purity = C.max(1)[used] / C.sum(1)[used]
    owned = owner[tpl] == clu
    n_u = np.bincount(clu, minlength=nU)
    borrowed_share = 1 - np.bincount(clu, weights=owned, minlength=nU) / np.maximum(n_u, 1)
    n_owned_templates = np.bincount(owner[owner >= 0], minlength=nU)

    # borrowed spikes: similarity of the unit to the template's owner, and distance between their peaks
    Wn = Wall.reshape(nU, -1).astype(np.float64)
    Wn /= np.linalg.norm(Wn, axis=1, keepdims=True) + 1e-12
    peak = (Wall.astype(np.float64) ** 2).sum(-1).argmax(1)
    b = ~owned
    pair, cnt = np.unique(clu[b] * nU + owner[tpl[b]], return_counts=True)
    pu, po = pair // nU, pair % nU
    cos = (Wn[pu] * Wn[po]).sum(1)
    unit_cos = np.bincount(pu, weights=cnt * cos, minlength=nU) / np.maximum(np.bincount(pu, weights=cnt, minlength=nU), 1)
    pk, ck = np.unique(clu[b] * nT + tpl[b], return_counts=True)
    dist = np.hypot(xc[peak[pk // nT]] - xc[iU[pk % nT]], yc[peak[pk // nT]] - yc[iU[pk % nT]])

    # fit and the reference from owned spikes of eligible units at the same amplitude
    fit = score / np.maximum(amp, 1e-12)
    in_units = np.isin(clu, units)
    ref_mask = owned & in_units
    edges = np.quantile(amp[ref_mask], np.linspace(0, 1, N_AMP_BINS + 1))
    abin = np.clip(np.searchsorted(edges, amp, 'right') - 1, 0, N_AMP_BINS - 1)
    fit_ref = np.array([np.median(fit[ref_mask & (abin == i)]) for i in range(N_AMP_BINS)])
    g = fit_ref[abin] / np.maximum(fit, 1e-12)

    rows = []
    order = np.argsort(clu, kind='stable')
    bounds = np.searchsorted(clu[order], np.arange(nU + 1))
    for u in units:
        idx = order[bounds[u]:bounds[u + 1]]
        if idx.size == 0:
            continue
        s, o = score[idx], owned[idx]
        G = float(np.median(g[idx]))
        dens = np.sum(s < DENS * Th) / ((DENS - 1) * Th)
        extra = dens * Th * (1 - 1 / G)
        within = None
        if o.sum() >= 100 and (~o).sum() >= 100:  # fit of borrowed against owned spikes of the same unit, same amplitude
            r = []
            for i in np.unique(abin[idx]):
                m = abin[idx] == i
                if (m & o).sum() >= 10 and (m & ~o).sum() >= 10:
                    r.append((np.median(fit[idx][m & ~o]) / np.median(fit[idx][m & o]), (m & ~o).sum()))
            if r:
                r = np.array(r)
                within = float(np.average(r[:, 0], weights=r[:, 1]))
        rows.append({'unit': int(u), 'n': int(idx.size), 'borrowed_share': float(borrowed_share[u]),
                     'n_owned_templates': int(n_owned_templates[u]), 'median_amp': float(np.median(amp[idx])),
                     'median_fit': float(np.median(fit[idx])), 'G': G, 'extra_spikes': float(extra),
                     'n_near': int(np.sum(s < DENS * Th)),
                     'borrowed_cos': float(unit_cos[u]) if borrowed_share[u] > 0 else None,
                     'within_unit_fit_ratio': within, 'S_i': None if S_i is None else S_i[int(u)]})

    in_b = np.isin(clu[b], units)
    bu = np.isin(pu, units)
    bk = np.isin(pk // nT, units)
    return {
        'sort': str(sort), 'Th_learned': Th, 'n_units': int(nU), 'n_extraction_templates': int(nT),
        'n_templates_with_spikes': int(used.sum()), 'n_eligible': int(units.size),
        'template_purity_median': float(np.median(purity)),
        'n_spikes_eligible': int(in_units.sum()), 'borrowed_spike_fraction_eligible': float(b[in_units].mean()),
        'borrowed_cos_quantiles': dict(zip(['10', '25', '50', '75', '90'],
                                           np.quantile(np.repeat(cos[bu], cnt[bu]), [.1, .25, .5, .75, .9]).tolist())),
        'borrowed_peak_dist_um_quantiles': dict(zip(['10', '25', '50', '75', '90'],
                                                    np.quantile(np.repeat(dist[bk], ck[bk]), [.1, .25, .5, .75, .9]).tolist())),
        'fit_ref_by_amp': {'edges': edges.tolist(), 'median_fit_owned': fit_ref.tolist()},
        'median_fit_owned_eligible': float(np.median(fit[ref_mask])),
        'median_fit_borrowed_eligible': float(np.median(fit[b & in_units])) if in_b.any() else None,
        'runtime_s': {k: float(ops[k]) for k in ('runtime_st', 'runtime_clu', 'runtime_merge', 'runtime')},
        'units': rows,
    }


def summarise(r):
    U = r['units']
    n = np.array([u['n'] for u in U])
    bs = np.array([u['borrowed_share'] for u in U])
    orphan = np.array([u['n_owned_templates'] == 0 for u in U])
    amp = np.array([u['median_amp'] for u in U])
    G = np.array([u['G'] for u in U])
    extra = np.array([u['extra_spikes'] for u in U])
    S = np.array([u['S_i'] if u['S_i'] is not None else np.nan for u in U])
    wr = np.array([u['within_unit_fit_ratio'] for u in U if u['within_unit_fit_ratio'] is not None])
    rt = r['runtime_s']
    print(f"\n== {Path(r['sort']).name}: {r['n_units']} units, {r['n_extraction_templates']} extraction templates "
          f"({r['n_templates_with_spikes']} with spikes), {r['n_eligible']} eligible; template purity median "
          f"{r['template_purity_median']:.2f}")
    print(f"  a third round adds extract + cluster = {rt['runtime_st'] + rt['runtime_clu']:.0f} s to {rt['runtime']:.0f} s")
    print(f"  borrowed spikes: {r['borrowed_spike_fraction_eligible']:.1%} of eligible units' spikes")
    for lo, hi in ((0.5, 1.01), (0.9, 1.01)):
        m = (bs >= lo) & (bs < hi)
        line = f"  units with borrowed share >= {lo:.0%}: {m.mean():.1%} of units, {n[m].sum() / n.sum():.1%} of spikes"
        if np.isfinite(S).any():
            line += f", {np.nansum(S[m]) / np.nansum(S):.1%} of S"
        print(line)
    line = f"  orphans (own no template): {orphan.mean():.1%} of units, {n[orphan].sum() / n.sum():.1%} of spikes"
    if np.isfinite(S).any():
        line += f", {np.nansum(S[orphan]) / np.nansum(S):.1%} of S"
    print(line)
    t = np.quantile(amp, [1 / 3, 2 / 3])
    for lab, m in (('small', amp < t[0]), ('middle', (amp >= t[0]) & (amp < t[1])), ('large', amp >= t[1])):
        print(f"  {lab:6s} units (amp tertile): borrowed share >= 50% in {(bs[m] >= .5).mean():.1%}, orphans {orphan[m].mean():.1%}, "
              f"median G {np.median(G[m]):.3f}")
    q = r['borrowed_cos_quantiles']
    d = r['borrowed_peak_dist_um_quantiles']
    print(f"  borrowed spikes: cos(unit, template owner) quantiles 10/50/90 = {q['10']:.2f}/{q['50']:.2f}/{q['90']:.2f}; "
          f"peak distance 10/50/90 = {d['10']:.0f}/{d['50']:.0f}/{d['90']:.0f} um")
    print(f"  fit (score/amp): owned {r['median_fit_owned_eligible']:.3f}, borrowed {r['median_fit_borrowed_eligible']:.3f} (median)")
    if wr.size:
        print(f"  within-unit fit, borrowed/owned at same amplitude ({wr.size} units with both): "
              f"median {np.median(wr):.3f}, 10-90% {np.quantile(wr, .1):.3f}-{np.quantile(wr, .9):.3f}")
    hb, lb = bs >= .5, bs < .5
    print(f"  G median: borrowed-share>=50% units {np.median(G[hb]):.3f}, others {np.median(G[lb]):.3f}")
    print(f"  extra spikes (signed, rough): all units {extra.sum() / n.sum():+.2%} of spikes; "
          f"borrowed-share>=50% units {extra[hb].sum() / n[hb].sum():+.2%} of their spikes "
          f"({extra[hb].sum() / n.sum():+.2%} of all); others {extra[lb].sum() / n[lb].sum():+.2%}")
    # G calibrated by units that own their templates (borrowed share < 10%) of similar amplitude, since G
    # varies between units for reasons other than the template (it is above 1 in owned small units)
    n_near = np.array([u['n_near'] for u in U])
    dec = np.clip(np.searchsorted(np.quantile(amp, np.linspace(0, 1, 11)), amp, 'right') - 1, 0, 9)
    G_ref = np.array([np.median(G[(dec == i) & (bs < .1)]) for i in range(10)])
    Gc = G / G_ref[dec]
    extra_c = n_near / (DENS - 1) * (1 - 1 / Gc)
    own = bs < .1
    print(f"  calibrated G median: borrowed-share>=50% units {np.median(Gc[hb]):.3f}; extra spikes (calibrated): "
          f"borrowed-share>=50% units {extra_c[hb].sum() / n[hb].sum():+.2%} of their spikes ({extra_c[hb].sum() / n.sum():+.2%} of all); "
          f"owned (<10%) units {extra_c[own].sum() / n[own].sum():+.2%} (should be near 0)")
    bc = np.array([u['borrowed_cos'] if u['borrowed_cos'] is not None else np.nan for u in U])
    for lab, m in (('cos >= 0.95 (near-copies of the owner)', hb & (bc >= .95)), ('cos < 0.95', hb & (bc < .95))):
        line = f"  borrowed-share>=50% units with {lab}: {m.sum()} units, {n[m].sum() / n.sum():.1%} of spikes"
        if np.isfinite(S).any():
            line += f", {np.nansum(S[m]) / np.nansum(S):.1%} of S"
        print(line)


if __name__ == '__main__':
    names = sys.argv[1:] or ['kobe_H8_s1', 'kobe_default_s1', 'mec_H8_s1', 'mec_default_s1']
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        r = analyse(name)
        (OUT / f'borrowed_{name}.json').write_text(json.dumps(r, indent=1))
        summarise(r)
