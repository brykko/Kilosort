"""Audit of make_pc_features: are rows of pc_features.npy written with the
channels of the spike's own cluster?

No tF is needed. A synthetic tF with two "PCs" is built per spike s with
template t: PC0 = norm over time of template t on channel iC[k, t] (from
templates.npy, so the channel ranking inside make_pc_features is realistic),
PC1 = (channel id + 1) * 1e-3, a label small enough not to move the ranking.
After make_pc_features every slot's PC1 says which channel's data it carries
(0 = the zero padding get_data_cpu adds for channels outside the spike's
template). A slot is correct when it holds feature_ind[own cluster, k] if that
channel is in the template's neighbourhood, else 0.

Runs stock make_pc_features (kilosort/postprocessing.py) and the fixed
version below on the saved spike_detection_templates / spike_clusters / ops
of a sort. CPU only, read-only on the sort folder.

    python pc_features_audit.py RUN_DIR [RUN_DIR ...] [--out JSON]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kilosort.postprocessing import make_pc_features
from kilosort.clustering_qr import get_data_cpu, xy_templates
from kilosort.io import load_ops


def make_pc_features_fixed(ops, spike_templates, spike_clusters, tF):
    """make_pc_features with igood restricted to the spikes of cluster i.

    Each spike's row is then read once, in its template's channel order, and
    written once, in its own cluster's channel order. Channel ranking uses the
    cluster's own spikes only.
    """
    xy, iC = xy_templates(ops)
    n_templates = iC.shape[1]
    n_clusters = np.unique(spike_clusters).size
    n_chans = ops['nearest_chans']
    feature_ind = np.zeros((n_clusters, n_chans), dtype=np.uint32)

    def groups(labels):
        order = np.argsort(labels, kind='stable')
        bounds = np.concatenate(([0], np.cumsum(np.bincount(labels))))
        return order, bounds
    clu_order, clu_bounds = groups(spike_clusters)
    pid = torch.from_numpy(spike_templates).long()

    for i in np.unique(spike_clusters):
        # Spikes of cluster i, ascending (stable argsort keeps them sorted)
        igood = clu_order[clu_bounds[i]:clu_bounds[i+1]]
        iunq = np.unique(spike_templates[igood]).astype(int)
        ix = torch.from_numpy(np.zeros(n_templates, bool))
        ix[iunq] = True
        Xd, igood, ichan = get_data_cpu(
            ops, xy, iC, pid, tF, None, None,
            dmin=ops['dmin'], dminx=ops['dminx'], ix=ix, merge_dim=False,
            igood=torch.from_numpy(igood)
            )
        spike_mean = Xd.mean(0)
        chan_norm = torch.linalg.norm(spike_mean, dim=1)
        sorted_chans, ind = torch.sort(chan_norm, descending=True)
        tF[igood,:] = Xd[:, ind[:n_chans], :]
        feature_ind[i,:] = ichan[ind[:n_chans]].cpu().numpy()

    tF = torch.permute(tF, (0, 2, 1))
    return tF, feature_ind


LABEL = 1e-3


def label_tF(iC, spike_templates, amp):
    iCt = iC.numpy().T                                      # (n_t, nC)
    amp_t = np.take_along_axis(amp, iCt, 1)                 # (n_t, nC)
    tF = np.empty((spike_templates.size, iCt.shape[1], 2), np.float32)
    tF[:, :, 0] = amp_t[spike_templates]
    tF[:, :, 1] = (iCt[spike_templates] + 1) * LABEL
    return torch.from_numpy(tF)


def check(out, feature_ind, spike_clusters, spike_templates, iC):
    """Compare each slot with what pc_feature_ind of the spike's own cluster
    promises: the label of that channel if it is in the spike's template
    neighbourhood, else 0 (padding)."""
    got = np.rint(out[:, 1, :].numpy() / LABEL).astype(np.int64)  # (n, n_chans)
    iCt = iC.numpy().T                                          # (n_t, nC)
    n = got.shape[0]
    bad = np.zeros(got.shape, bool)
    lost = np.zeros(got.shape, bool)   # own channel has data, slot shows 0
    for a in range(0, n, 500_000):
        b = min(a + 500_000, n)
        ch = feature_ind[spike_clusters[a:b]].astype(np.int64)  # (m, n_chans)
        member = (iCt[spike_templates[a:b]][:, None, :] == ch[:, :, None]).any(2)
        want = np.where(member, ch + 1, 0)
        bad[a:b] = got[a:b] != want
        lost[a:b] = member & (got[a:b] == 0)
    return {
        'rows_wrong': float(bad.any(1).mean()),
        'slots_wrong': float(bad.mean()),
        'slot0_wrong': float(bad[:, 0].mean()),
        'slots_own_data_zeroed': float(lost.mean()),
        'slots_other_channel': float((bad & (got != 0)).mean()),
    }


def write_counts(spike_templates, spike_clusters):
    """How many clusters write each spike's row under the stock code, and
    which cluster writes it last (clusters run in ascending id)."""
    n_t = spike_templates.max() + 1
    pairs = np.unique(np.stack([spike_clusters, spike_templates], 1), axis=0)
    n_writers_t = np.bincount(pairs[:, 1], minlength=n_t)
    last_writer_t = np.full(n_t, -1)
    np.maximum.at(last_writer_t, pairs[:, 1], pairs[:, 0])
    first_writer_t = np.full(n_t, np.iinfo(np.int64).max)
    np.minimum.at(first_writer_t, pairs[:, 1], pairs[:, 0])
    nw = n_writers_t[spike_templates]
    return {
        'n_spikes': int(spike_templates.size),
        'n_clusters': int(np.unique(spike_clusters).size),
        'n_templates_used': int((n_writers_t > 0).sum()),
        'templates_shared': float((n_writers_t > 1).sum() / (n_writers_t > 0).sum()),
        'clusters_mixing_templates': float(np.mean(
            np.bincount(pairs[:, 0]) [np.unique(spike_clusters)] > 1)),
        'spikes_written_more_than_once': float((nw > 1).mean()),
        'spikes_last_writer_not_own': float(
            (last_writer_t[spike_templates] != spike_clusters).mean()),
        'spikes_first_and_only_writer_own': float(
            ((nw == 1) & (first_writer_t[spike_templates] == spike_clusters)).mean()),
        'mean_writes_per_spike': float(nw.mean()),
    }


def audit(run_dir):
    run_dir = Path(run_dir)
    st = np.load(run_dir / 'spike_detection_templates.npy').astype(np.int64)
    clu = np.load(run_dir / 'spike_clusters.npy').astype(np.int64)
    ops = load_ops(run_dir / 'ops.npy', device=torch.device('cpu'))
    res = {'run': str(run_dir), 'nearest_chans': int(ops['nearest_chans'])}
    res.update(write_counts(st, clu))
    _, iC = xy_templates(ops)
    # templates.npy holds one template per final cluster. Peak channel of a
    # cluster (what slot 0 of pc_feature_ind should be) is its argmax; a
    # detection template's amplitude is the spike-weighted mean over the
    # clusters its spikes went to.
    amp_clu = np.linalg.norm(np.load(run_dir / 'templates.npy'), axis=1)
    peak = amp_clu.argmax(1)
    cnt = np.zeros((iC.shape[1], amp_clu.shape[0]))
    np.add.at(cnt, (st, clu), 1)
    amp = (cnt / np.maximum(cnt.sum(1, keepdims=True), 1)) @ amp_clu
    ucl = np.unique(clu)
    for name, fn in [('stock', make_pc_features), ('fixed', make_pc_features_fixed)]:
        out, find = fn(ops, st, clu, label_tF(iC, st, amp))
        res[name] = check(out, find, clu, st, iC)
        res[name]['clusters_slot0_is_peak'] = float(
            (find[ucl, 0] == peak[ucl]).mean())
        res[name + '_feature_ind'] = find
    fs, ff = res.pop('stock_feature_ind'), res.pop('fixed_feature_ind')
    res['feature_ind_rows_differ'] = float((fs[ucl] != ff[ucl]).any(1).mean())
    res['feature_ind_slot0_differ'] = float((fs[ucl, 0] != ff[ucl, 0]).mean())
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--out')
    a = p.parse_args()
    torch.set_grad_enabled(False)
    results = []
    for r in a.runs:
        res = audit(r)
        print(json.dumps(res, indent=1), flush=True)
        results.append(res)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))
