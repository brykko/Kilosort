"""One CPU pass over a recording, preprocessed as KS4 preprocessed it (filter, CAR, whitening, drift shift from the
sort's ops.npy), giving each unit's spikes in KS4's feature space: the 6 PCs (wPCA) on the 32 channels nearest the
unit's peak channel. Features are read from the conv1d of the batch with wPCA, as run_matching does; they're of the
batch as it is, not of the residual, so overlapping spikes are not subtracted (KS4's tF has them subtracted).

For each label set (a saved sort at some round and seed, all read through this pass's preprocessing), per unit:
    n, sum, n/sum of two random halves, of the first and last epochs and of those epochs' random halves, and the
    summed squared norm of the features on 10 and 32 channels (per-spike scatter).
For the label sets in --reservoir, also a random sample (seed 0) of up to --res spikes per eligible unit, on 10
channels (KS4's nearest_chans), with times, for the sample-complexity curves.

Writes <out>/<name>.npz per label set; reservoirs as <name>_res.npy + <name>_res_idx.npz.

    python snippets.py --family kobe_H8 [--batches first:last] [--threads N]
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.functional import conv1d  # noqa: E402

from common import BATCH, DUR, NT, NT0, OUT, SORTS, IT, Sort, sort_dir  # noqa: E402
from kilosort.io import BinaryFiltered, load_ops  # noqa: E402

KCH = 32
KFEAT = 10
EPOCH_S = 300.0
LW = Path('C:/data/ks4_tuning/exploratory/long_data/windows')


def family(name):
    """(ops folder, label sets {name: (folder, ex, setting)}, reservoir names, first epoch end, last epoch start, batches)"""
    ex, setting = name.split('_', 1)
    if setting == '2h':
        d = SORTS / f'{ex}_2h_repeat_s1'
        o = np.load(d / 'ops.npy', allow_pickle=True).item()
        nb = int(o['Nbatches'])
        return d, {f'{ex}_2h_r4': (d, ex, 'H8')}, [], 1200.0, nb * 2.0 - 1200.0, [*range(0, 600), *range(nb - 600, nb)]
    if setting == 'H8':
        sets = {f'{ex}_H8_s{s}_r{r}': (sort_dir(ex, 'H8', s, r), ex, 'H8') for s, r in ((1, 2), (1, 3), (1, 4), (2, 2), (2, 4))}
        res = [f'{ex}_H8_s1_r2', f'{ex}_H8_s1_r4']
        return sort_dir(ex, 'H8', 1, 4), sets, res, EPOCH_S, DUR - EPOCH_S, None
    sets = {f'{ex}_default_s{s}_r{r}': (sort_dir(ex, 'default', s, r), ex, 'default') for s, r in ((1, 2), (1, 3), (2, 3))}
    return sort_dir(ex, 'default', 1, 3), sets, [f'{ex}_default_s1_r3'], EPOCH_S, DUR - EPOCH_S, None


class Labels:
    def __init__(self, name, d, ex, setting, ops, first_end, last_start, res, rng):
        s = Sort(d, ex, setting, clean=not setting == 'none')
        self.name, self.first_end, self.last_start = name, first_end * 30000, last_start * 30000
        self.t, c = s.t[s.keep], s.clu[s.keep]
        self.c = c
        U = np.load(Path(d) / 'Wall.npy')
        peak = (U ** 2).sum(-1).argmax(1)
        xc, yc = np.asarray(ops['xc']), np.asarray(ops['yc'])
        ds = (xc - xc[:, None]) ** 2 + (yc - yc[:, None]) ** 2
        self.chans = torch.from_numpy(np.argsort(ds, 0, kind='stable')[:KCH][:, peak].T.astype(np.int64))  # units x KCH
        self.n_units = U.shape[0]
        self.half = rng.integers(0, 2, self.t.size).astype(np.int64)
        nu, shape = self.n_units, (self.n_units, KCH, 6)
        self.sums = {k: torch.zeros(shape, dtype=torch.float64) for k in
                     ('all', 'h0', 'h1', 'first', 'last', 'first0', 'first1', 'last0', 'last1')}
        self.ns = {k: np.zeros(nu, np.int64) for k in self.sums}
        self.sq10, self.sq32 = np.zeros(nu), np.zeros(nu)
        self.res = None
        self.b = np.searchsorted(self.t, np.arange(0, 10 ** 12, BATCH)[:int(self.t[-1] // BATCH + 3)])
        if res:
            elig = np.array(sorted(s.eligible()))
            o = np.argsort(c, kind='stable')
            lim = np.searchsorted(c[o], np.arange(nu + 1))
            pick = []
            for u in elig:
                idx = o[lim[u]:lim[u + 1]]
                pick.append(rng.permutation(idx)[:res])
            lens = np.array([p.size for p in pick])
            self.res_unit = np.repeat(elig, lens)
            pick = np.concatenate(pick)
            ord2 = np.argsort(pick)
            self.res_idx, self.res_unit = pick[ord2], self.res_unit[ord2]    # sorted by spike index
            self.res_feat = np.lib.format.open_memmap(OUT / f'{name}_res.npy', 'w+', np.float32,
                                                      (pick.size, KFEAT, 6))
            self.res = True

    def add(self, B, ib, batches_all):
        i0, i1 = self.b[ib], self.b[ib + 1]
        if i1 <= i0:
            return
        t, c = self.t[i0:i1], self.c[i0:i1]
        iX = torch.from_numpy(t - ib * BATCH + NT + 10)
        cu = torch.from_numpy(c)
        F = B[self.chans[cu].unsqueeze(1).expand(-1, 6, -1), torch.arange(6).view(1, 6, 1), iX.view(-1, 1, 1)]
        F = F.permute(0, 2, 1).double()                                       # spikes x KCH x 6
        fn = (F ** 2).sum(2)
        self.sq10 += np.bincount(c, fn[:, :KFEAT].sum(1).numpy(), minlength=self.n_units)
        self.sq32 += np.bincount(c, fn.sum(1).numpy(), minlength=self.n_units)

        def add(key, mask):
            if mask is None:
                self.sums[key].index_add_(0, cu, F)
                self.ns[key] += np.bincount(c, minlength=self.n_units)
            elif mask.any():
                m = torch.from_numpy(mask)
                self.sums[key].index_add_(0, cu[m], F[m])
                self.ns[key] += np.bincount(c[mask], minlength=self.n_units)
        add('all', None)
        h = self.half[i0:i1]
        add('h0', h == 0)
        add('h1', h == 1)
        fi, la = t < self.first_end, t >= self.last_start
        add('first', fi)
        add('last', la)
        add('first0', fi & (h == 0))
        add('first1', fi & (h == 1))
        add('last0', la & (h == 0))
        add('last1', la & (h == 1))
        if self.res:
            a, e = np.searchsorted(self.res_idx, [i0, i1])
            if e > a:
                self.res_feat[a:e] = F[torch.from_numpy(self.res_idx[a:e] - i0), :KFEAT].numpy().astype(np.float32)

    def save(self):
        np.savez(OUT / f'{self.name}.npz', chans=self.chans.numpy(), sq10=self.sq10, sq32=self.sq32,
                 **{f'n_{k}': v for k, v in self.ns.items()}, **{f's_{k}': v.numpy() for k, v in self.sums.items()})
        if self.res:
            self.res_feat.flush()
            np.savez(OUT / f'{self.name}_res_idx.npz', unit=self.res_unit, t=self.t[self.res_idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', required=True)
    p.add_argument('--res', type=int, default=4000)
    p.add_argument('--threads', type=int, default=6)
    p.add_argument('--test', type=int, default=0, help='only this many batches, nothing saved')
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    ops_dir, sets, res, first_end, last_start, batches = family(a.family)
    device = torch.device('cpu')
    ops = load_ops(ops_dir / 'ops.npy', device=device)
    fn = ops['filename']
    fn = fn[0] if isinstance(fn, (list, tuple)) else fn
    bfile = BinaryFiltered(fn, ops['n_chan_bin'], fs=ops['fs'], NT=ops['batch_size'], nt=ops['nt'], nt0min=ops['nt0min'],
                           chan_map=ops['probe']['chanMap'], hp_filter=ops['fwav'], whiten_mat=ops['Wrot'],
                           dshift=ops['dshift'], device=device, do_CAR=ops['do_CAR'],
                           artifact_threshold=ops['artifact_threshold'], invert_sign=ops['invert_sign'],
                           dtype=ops['data_dtype'], tmin=ops['tmin'], tmax=ops['tmax'], shift=ops['shift'],
                           scale=ops['scale'])
    W = ops['wPCA'].contiguous().cpu().float()
    rng = np.random.default_rng(0)
    L = [Labels(n, *v, ops, first_end, last_start, a.res if n in res else 0, rng) for n, v in sets.items()]
    batches = list(range(bfile.n_batches)) if batches is None else batches
    if a.test:
        batches = batches[300:300 + a.test]
    t0 = time.time()
    for k, ib in enumerate(batches):
        X = bfile.padded_batch_to_torch(ib, ops)
        B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=NT // 2)       # chans x 6 x time
        for lab in L:
            lab.add(B, ib, batches)
        if k % 50 == 0:
            print(f'{a.family}: batch {k}/{len(batches)}, {time.time() - t0:.0f} s', flush=True)
    if a.test:
        wall = np.load(sets[L[-1].name][0] / 'Wall.npy')
        lab = L[-1]
        m = lab.ns['all'] > 50
        s = lab.sums['all'].numpy()[m] / lab.ns['all'][m, None, None]
        w = np.stack([wall[u, lab.chans[u].numpy()] for u in np.flatnonzero(m)])
        c = (s * w).sum((1, 2)) / np.sqrt((s ** 2).sum((1, 2)) * (w ** 2).sum((1, 2)))
        print('check: cosine of raw mean to Wall, quantiles .1 .5 .9', np.quantile(c, [.1, .5, .9]), 'units', m.sum())
        print('norms ratio raw/Wall median', np.median(np.sqrt((s ** 2).sum((1, 2)) / (w ** 2).sum((1, 2)))))
        print(f'{(time.time() - t0) / a.test:.2f} s per batch')
        return
    for lab in L:
        lab.save()
    print(f'{a.family} done, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
