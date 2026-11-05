"""The size of the template change that a round actually makes. Round 2 extracts with U (first-pass cluster means, rebuilt
in s1_dictionary.py); round 3 extracts with templates built from round 2's clusters. For each template of U, 1 - maximum-
lag correlation with the round-2 Wall of its owner (the unit that got most of its spikes), and with the Wall of the
unit it best matches. Weighted by spikes as well as by template. Then the residual left by subtracting a spike of norm a
with a template at correlation r is a*sqrt(2(1-r)): compared with the matching threshold, this says how much subtraction
could change. Seed 1, H8."""
from common import *  # noqa
import torch  # noqa: E402
from kilosort import template_matching as tm  # noqa: E402

torch.set_num_threads(3)
for ex in EXS:
    d = folder(ex, 'H8', 2, 1)
    U = torch.load(OUT / f'U_r2_{ex}_s1.pt').float()
    Wall = torch.from_numpy(np.load(d / 'Wall.npy')).float()
    tpl = np.load(d / 'spike_detection_templates.npy').ravel().astype(np.int64)
    clu = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
    amp = np.load(d / 'amplitudes.npy').ravel()
    ops = dict(ops_of(d))
    ops['wPCA'] = torch.as_tensor(tonp(ops['wPCA'])).float()
    owner, _, _, _ = owners(tpl, clu)
    r = tm.max_lag_correlation(U.transpose(1, 2), Wall, ops, block=24)
    idx = np.arange(U.shape[0])
    ok = owner[idx] >= 0
    r_own = r[idx[ok], owner[idx[ok]]]
    r_best = r[idx[ok]].max(1)
    w = np.bincount(tpl, minlength=U.shape[0])[idx[ok]].astype(float)
    wq = lambda x, q: np.sort(x)[np.searchsorted(np.cumsum(w[np.argsort(x)]) / w.sum(), q)]  # noqa: E731
    d_own = 1 - r_own
    d_best = 1 - r_best
    a = np.median(amp)
    print(f'{ex}: templates {ok.sum()}; 1-r to owner, median {np.median(d_own):.4f}, q75 {np.quantile(d_own, .75):.4f}, q90 {np.quantile(d_own, .9):.4f}; '
          f'spike-weighted median {wq(d_own, .5):.4f}, q90 {wq(d_own, .9):.4f}; to best unit, median {np.median(d_best):.4f}; '
          f'spike-weighted {wq(d_best, .5):.4f}; owner is best unit for {np.mean(r_own >= r_best - 1e-6):.3f}', flush=True)
    for q in (.5, .9):
        x = wq(d_own, q)
        print(f'   residual of a median spike (feature norm {a:.1f}) at spike-weighted q{q}: {a * (2 * x)**.5:.2f}; of a spike of norm 50: {50 * (2 * x)**.5:.2f}; Th_learned {ops["Th_learned"]}')
