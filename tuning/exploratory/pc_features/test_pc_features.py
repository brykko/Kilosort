"""Tests for make_pc_features (stock) and make_pc_features_fixed.

A row of pc_features is correct when, for each slot k, it holds the spike's
original features on channel feature_ind[own cluster, k], or zeros if that
channel is outside the spike's template neighbourhood.

    python tuning/exploratory/pc_features/test_pc_features.py   (or pytest)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from pc_features_audit import make_pc_features_fixed           # noqa: E402
from kilosort.postprocessing import make_pc_features            # noqa: E402


def toy(seed=0, n_spikes=400, shared=True):
    """8 channels on a line 20 um apart, one template per channel, 3 nearest
    channels per template. With shared=True templates 2-5 are split between
    two clusters each, as KS4's final clustering does."""
    rng = np.random.default_rng(seed)
    n_chan, nC, n_pcs = 8, 3, 4
    yc = np.arange(n_chan) * 20.0
    xc = np.zeros(n_chan)
    d = np.abs(yc[:, None] - yc[None, :])
    iCC = torch.from_numpy(np.argsort(d, axis=0, kind='stable')[:nC]).long()
    ops = {'iU': torch.arange(n_chan), 'iCC': iCC, 'xc': xc, 'yc': yc,
           'nearest_chans': nC, 'dmin': 20, 'dminx': 32}
    st = rng.integers(0, n_chan, n_spikes)
    if shared:
        # cluster = template // 2, except half of templates 2-5's spikes go
        # to the neighbouring cluster
        clu = st // 2
        flip = (st >= 2) & (st <= 5) & (rng.random(n_spikes) < 0.5)
        clu[flip] = np.where(st[flip] % 2 == 0, clu[flip] - 1, clu[flip] + 1)
    else:
        clu = st // 2
    # Per-channel scale so each cluster's peak channel is well defined
    tF = torch.from_numpy(rng.normal(size=(n_spikes, nC, n_pcs)).astype(np.float32))
    tF *= torch.tensor([3.0, 1.5, 0.5])[None, :, None]
    return ops, st.astype(np.int64), clu.astype(np.int64), tF


def expected(ops, st, clu, tF0, feature_ind):
    iC = ops['iCC'][:, ops['iU']].numpy()
    n, nC, n_pcs = tF0.shape
    out = np.zeros((n, n_pcs, feature_ind.shape[1]), np.float32)
    for s in range(n):
        for k, ch in enumerate(feature_ind[clu[s]]):
            pos = np.nonzero(iC[:, st[s]] == ch)[0]
            if pos.size:
                out[s, :, k] = tF0[s, pos[0], :]
    return out


def errors(fn, ops, st, clu, tF0):
    out, find = fn(ops, st, clu, tF0.clone())
    return np.abs(out.numpy() - expected(ops, st, clu, tF0.numpy(), find)).max(), find


def test_fixed_rows_match_own_cluster():
    ops, st, clu, tF0 = toy(shared=True)
    err, _ = errors(make_pc_features_fixed, ops, st, clu, tF0)
    assert err == 0


def test_stock_rows_wrong_when_templates_shared():
    # Documents the bug: rows of spikes whose template is shared between
    # clusters are reread after another cluster rewrote them.
    ops, st, clu, tF0 = toy(shared=True)
    err, _ = errors(make_pc_features, ops, st, clu, tF0)
    assert err > 0


def test_fixed_equals_stock_without_sharing():
    # One cluster per template set: the fix changes nothing.
    ops, st, clu, tF0 = toy(shared=False)
    a, fa = make_pc_features(ops, st, clu, tF0.clone())
    b, fb = make_pc_features_fixed(ops, st, clu, tF0.clone())
    assert torch.equal(a, b) and np.array_equal(fa, fb)
    err, _ = errors(make_pc_features, ops, st, clu, tF0)
    assert err == 0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
