"""T2 of the interleaving audit: could a residual-based birth see units absorbed by another template? A spike of a unit
caught by a template at cosine c leaves, after the peel, a signal of norm s * sqrt(1 - c^2), with s the spike's own signal
norm. A learner that detects in the residual sees it only if that exceeds the threshold.

Round 2 sorts, H8 and default, seeds 1-3. Same rule for all eligible units (borrowed.json's rows, the harness's eligible set);
none is selected further. Reported for three groups of spikes: orphans' spikes (units owning no template, as in
borrowed_templates.py), all borrowed spikes of eligible units, and owned spikes as a control.

Quantities:
- c: maximum-lag correlation between the unit's final template (Wall) and the catching template's shape. Extraction
  templates aren't saved, so the shape is the spike-weighted mean of Wall over the units it caught, for every seed. At seed 1 of
  H8 the rebuilt U of round 2 is used as well, as a check on that proxy.
- s, two estimates: per spike, score / c (score is the projection on the template; it is conditioned on passing Th, which
  biases s up); per unit, |Wall|, the mean signal norm.
- visible: residual signal r above Th; 'with noise': the mean of Phi(r - Th), matched-filter noise of SD 1.

r above Th is an upper bound on what a real residual pass sees, since it assumes a filter matched to the residual's own shape.
'universal' applies a factor 0.85, the median fit of spikes to the universal bank (empirics section 5), as a rough lower figure.

    python t2_orphans.py
"""
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
WT = Path('C:/code/github/Kilosort/.claude/worktrees/ks4-round3')
sys.path.insert(0, str(WT))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402
from scipy.stats import norm  # noqa: E402
from kilosort import template_matching as tm  # noqa: E402

O = Path('C:/data/ks4_tuning/exploratory/iteration/sorts')
UD = Path(r'C:/Users/richa/AppData/Local/Temp/claude/8c1cec062ef79d80/45eb8e85-37d5-451f-bed4-1560d425db7e/scratchpad/skeptic_out')
OUT = Path('C:/data/ks4_tuning/exploratory/interleaving/checks')
UNIV = 0.85
torch.set_num_threads(4)


def tonp(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def group(r, Th, w=None):
    if r.size == 0:
        return None
    return {'n': int(r.size), 'median_r': float(np.median(r)), 'q90_r': float(np.quantile(r, .9)),
            'frac_above': float(np.mean(r > Th)), 'frac_above_noise': float(np.mean(norm.cdf(r - Th))),
            'frac_above_universal': float(np.mean(UNIV * r > Th))}


def analyse(d, shape_mode):
    clu = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
    tpl = np.load(d / 'spike_detection_templates.npy').ravel().astype(np.int64)
    score = np.load(d / 'score.npy').ravel().astype(np.float64)
    Wall = np.load(d / 'Wall.npy').astype(np.float32)
    ops = dict(np.load(d / 'ops.npy', allow_pickle=True).item())
    ops['wPCA'] = torch.as_tensor(tonp(ops['wPCA'])).float()
    Th = float(ops['Th_learned'])
    rows = json.loads((d / 'borrowed.json').read_text())['units']
    eligible = np.array([r['unit'] for r in rows])
    orphan = np.array([r['unit'] for r in rows if r['n_owned_templates'] == 0])
    nU, nT = Wall.shape[0], int(tpl.max()) + 1
    C = np.bincount(tpl * nU + clu, minlength=nT * nU).reshape(nT, nU)
    used = C.sum(1) > 0
    owner = np.where(used, C.argmax(1), -1)
    if shape_mode == 'proxy':
        P = sp.csr_matrix(C.astype(np.float32))
        P = sp.diags(1 / np.maximum(np.asarray(P.sum(1)).ravel(), 1)) @ P
        Ut = torch.from_numpy((P @ Wall.reshape(nU, -1)).reshape(nT, *Wall.shape[1:]).astype(np.float32))
    else:
        Ut = torch.load(UD / shape_mode, weights_only=False).float().transpose(1, 2)
        assert Ut.shape[0] >= nT
        Ut = Ut[:nT]
    rmat = tm.max_lag_correlation(torch.from_numpy(Wall), Ut, ops, block=64)      # nU x nT
    c = np.clip(rmat[clu, tpl].astype(np.float64), 1e-3, 1)
    fac = np.sqrt(1 - c ** 2)
    r_spike = score / c * fac
    norms = np.sqrt((Wall.astype(np.float64) ** 2).sum((1, 2)))
    r_unit = norms[clu] * fac
    is_e = np.isin(clu, eligible)
    is_o = np.isin(clu, orphan)
    bor = owner[tpl] != clu
    out = {'sort': str(d), 'shape': shape_mode, 'Th_learned': Th, 'n_eligible': int(eligible.size), 'n_orphans': int(orphan.size),
           'median_c': {'orphan': float(np.median(c[is_o])), 'borrowed': float(np.median(c[is_e & bor])), 'owned': float(np.median(c[is_e & ~bor]))}}
    for name, m in (('orphan', is_o), ('borrowed', is_e & bor), ('owned', is_e & ~bor)):
        out[name] = {'per_spike': group(r_spike[m], Th), 'per_unit': group(r_unit[m], Th)}
    # per orphan unit: share of its spikes above threshold (per-unit signal)
    a = (r_unit > Th)
    per = np.bincount(clu[is_o], weights=a[is_o], minlength=nU)[orphan] / np.maximum(np.bincount(clu[is_o], minlength=nU)[orphan], 1)
    out['orphan_units'] = {'median_share_visible': float(np.median(per)), 'share_units_half_visible': float(np.mean(per >= .5))}
    return out


if __name__ == '__main__':
    res = []
    for ex in ('kobe', 'mec'):
        for g in ('H8', 'default'):
            pre = '' if g == 'H8' else 'default_'
            for s in (1, 2, 3):
                d = O / f'{ex}_{pre}repeat_s{s}' / 'round2'
                modes = ['proxy'] + ([f'U_r2_{ex}_s1.pt'] if g == 'H8' and s == 1 else [])
                for m in modes:
                    x = analyse(d, m)
                    x.update(ex=ex, set=g, seed=s)
                    res.append(x)
                    o = x['orphan']
                    print(f"{ex} {g} s{s} {m[:5]}: orphans {x['n_orphans']}/{x['n_eligible']}, Th {x['Th_learned']:.0f}, c {x['median_c']['orphan']:.3f}; "
                          f"orphan spikes above Th: spike {o['per_spike']['frac_above']:.3f}, unit {o['per_unit']['frac_above']:.3f}, noise {o['per_unit']['frac_above_noise']:.3f}, "
                          f"univ {o['per_unit']['frac_above_universal']:.3f}; borrowed {x['borrowed']['per_unit']['frac_above']:.3f}; owned {x['owned']['per_unit']['frac_above']:.3f}", flush=True)
                    (OUT / 't2_orphans.json').write_text(json.dumps(res, indent=1))
    print('done', flush=True)
