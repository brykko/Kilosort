"""How many correctly labelled spikes before a template can be told from its nearest neighbour.

All eligible units of the final round (seed 1). Nearest neighbour: the other unit (any unit of the sort, eligible or not)
of largest cosine at lag 0, full template, Wall @ wPCA. Distance d2 = |Ta - Tb|^2 / |Ta|^2. Two sample means differ reliably
(2 sd of the difference, equal n) when n >= 8 s2 / |Ta - Tb|^2; s2 per spike from m34_samples.py, 10 channels, batch features.
Compare with the batches of m34_samples. Lower bound: assumes labels are already right.

    python m3_neighbour.py
"""
import numpy as np

from common import OUT, sort_dir

for name, ex, setting, r in (('kobe_H8_s1_r4', 'kobe', 'H8', 4), ('mec_H8_s1_r4', 'mec', 'H8', 4),
                             ('kobe_default_s1_r3', 'kobe', 'default', 3), ('mec_default_s1_r3', 'mec', 'default', 3)):
    d = sort_dir(ex, setting, 1, r)
    z = np.load(OUT / f'm3_{name}.npz')
    w = np.asarray(np.load(d / 'ops.npy', allow_pickle=True).item()['wPCA'])
    T = (np.load(d / 'Wall.npy') @ w).astype(np.float32)
    T = T.reshape(T.shape[0], -1)
    nrm = np.sqrt((T ** 2).sum(1))
    N = T / np.maximum(nrm, 1e-6)[:, None]
    C = N @ N[z['unit']].T                                  # all units x eligible
    C[z['unit'], np.arange(z['unit'].size)] = -1
    nn = C.argmax(0)
    c = C.max(0)
    a = T[z['unit']]
    d2 = ((a - T[nn]) ** 2).sum(1) / (a ** 2).sum(1)
    shape = 2 * (1 - c)
    ns = 8 * z['sig2'] / z['T2'] / d2
    r_ = z['rate']
    print(name, 'NN cosine .1/.5/.9', np.quantile(c, [.1, .5, .9]).round(3), 'd .5', np.sqrt(np.median(d2)).round(3),
          'n_sep .1/.5/.9', np.quantile(ns, [.1, .5, .9]).round(0), 'batches', np.quantile(ns / (2 * r_), [.1, .5, .9]).round(1),
          'rate terciles median batches', [round(float(np.median((ns / (2 * r_))[z['rq'] == i])), 1) for i in range(3)], flush=True)
