"""How many spikes a unit has in a window of a given length: what clustering in small batches would see.
Scored units of the H8 round-4 sort, seed 1."""
from common import *  # noqa

for ex in EXS:
    d = folder(ex, 'H8', 4, 1)
    Si = S_i(d, ex)
    clu = np.load(d / 'spike_clusters.npy').ravel()
    ids = np.array(sorted(Si))
    n = np.bincount(clu, minlength=ids.max() + 1)[ids].astype(float)
    S = np.array([Si[i] for i in ids])
    rate = n / DUR
    nsp = clu.size
    print(ex, 'scored units', ids.size, 'S', round(S.sum(), 1), 'spikes', nsp, 'median rate', round(np.median(rate), 2),
          'q10,q25', np.round(np.quantile(rate, [.1, .25]), 2))
    for T in (2, 10, 30, 60, 120, 300, 600):
        for need in (30, 100):
            ok = rate * T >= need
            print(f'  window {T:4d} s, >= {need:3d} spikes expected: {ok.mean():.1%} of units, '
                  f'{S[ok].sum() / S.sum():.1%} of S')
    print('  spikes per 2 s batch', nsp / (DUR / 2))
