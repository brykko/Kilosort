"""Cross-check of the sample complexity in KS4's own features: tF of the round-2 extraction (stage cache), which has
overlapping spikes subtracted, against the batch features of snippets.py, which haven't.

All eligible units of round 2 (seed 1). A spike's tF is on the 10 channels nearest its extraction template's peak, so a
unit's spikes can sit on different channel sets; each unit uses the spikes of its most common set (bias: fewer spikes,
those best matched). Spikes are linked to the saved sort by (time, template); those the merge shifted in time are lost
(reported). Per unit: s2 / |T|^2 (scatter over signal, debiased), compared with the same ratio from the batch features,
and n*(0.95), n*(0.99).

    python m3_tf_check.py
"""
import gc
import json

import numpy as np
import torch

from common import IT, OUT, Sort, sort_dir

CS = (0.95, 0.99)


def run(ex):
    d = sort_dir(ex, 'H8', 1, 2)
    p = json.loads((sort_dir(ex, 'H8', 1, 4) / 'provenance.json').read_text())['cache']
    c = torch.load(IT / f"stage_cache/B_{p['B']}/extract.pt", map_location='cpu', weights_only=False)
    st, tF = c['out'][0], c['out'][1].numpy()
    iU, iCC = c['ops']['iU'].cpu().numpy(), c['ops']['iCC'].cpu().numpy()
    del c
    gc.collect()
    s = Sort(d, ex, 'H8')
    tpl = np.load(d / 'spike_detection_templates.npy').ravel().astype(np.int64)[s.order]
    t, cl = s.t[s.keep], s.clu[s.keep]
    tp = tpl[s.keep]
    M = int(tpl.max()) + 2
    key_e = st[:, 0].astype(np.int64) * M + st[:, 1].astype(np.int64)
    oe = np.argsort(key_e)
    key = t * M + tp
    i = np.searchsorted(key_e[oe], key)
    i = np.minimum(i, oe.size - 1)
    ok = key_e[oe][i] == key
    idx = oe[i]
    print(ex, 'linked', ok.mean(), flush=True)
    z = np.load(OUT / f'{ex}_H8_s1_r2.npz')
    peak = iU[tp]
    rows = []
    for u in sorted(s.eligible()):
        m = (cl == u) & ok
        if m.sum() < 20:
            continue
        pk = np.bincount(peak[m]).argmax()
        j = idx[m][peak[m] == pk]
        x = tF[j].astype(np.float64)
        N = x.shape[0]
        mean = x.mean(0)
        sig2 = (x ** 2).sum((1, 2)).mean() - (mean ** 2).sum()
        T2 = (mean ** 2).sum() - sig2 / N
        Nz = z['n_all'][u]
        Mz = z['s_all'][u, :10] / Nz
        sz = z['sq10'][u] / Nz - (Mz * Mz).sum()
        Tz = (Mz * Mz).sum() - sz / Nz
        rows.append((u, N, Nz, sig2, T2, sz, Tz))
    np.save(OUT / f'm3tf_{ex}.npy', np.array(rows))
    r = np.array(rows)
    a, b = r[:, 3] / r[:, 4], r[:, 5] / r[:, 6]
    print(ex, 'units', len(rows), 'share of unit spikes used', np.median(r[:, 1] / r[:, 2]), 's2/T2 tF median', np.median(a), 'batch', np.median(b),
          'ratio', np.median(a / b), flush=True)


if __name__ == '__main__':
    for ex in ('kobe', 'mec'):
        run(ex)
        gc.collect()
