"""T1 of the interleaving audit: what a round gives with no fresh clustering. Each saved round sort is relabelled by
extraction template (spike_detection_templates), so a spike's unit is the template that caught it, not its graph cluster.

Variants, written to C:/data/ks4_tuning/exploratory/interleaving/checks/variants/{ex}_{set}_r{k}_s{s}_{mode}:
- T: template labels;
- TM: template labels after KS4's final merge (merging_function, same thresholds as the sort), the fair match for a
  saved sort, which has had that merge;
- G: the sort's own labels through the same writer, as a control: it must reproduce the sort's known score.

Round k+1's templates are built from round k's clusters, so T at round 3 keeps round 2's partition and repeats only the
extraction: TM(r3) - G(r2) is the gain without re-clustering.

Same rule for every round, seed and setting; no unit is selected. Spikes are the sort's output spikes (after KS4's own
duplicate removal, whereas the sort's merge ran before it; a small difference). Extraction templates aren't saved, so a
variant's template is the spike-count-weighted mean of the final units' templates (templates.npy for the waveform
screen, Wall.npy for the merge's correlations). That mean blurs a template that fits a subgroup of a unit; it affects only
the screen and the merge's shape test. The merge's lag correction shifts spike times as in KS4; it rolls Wall only, since
tF isn't saved.

    python t1_variants.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.dont_write_bytecode = True
WT = Path('C:/code/github/Kilosort/.claude/worktrees/ks4-round3')
sys.path.insert(0, str(WT))

import torch  # noqa: E402
from kilosort import template_matching as tm  # noqa: E402

O = Path('C:/data/ks4_tuning/exploratory/iteration/sorts')
D = Path('C:/data/ks4_tuning/exploratory/interleaving/checks/variants')
EXS = ('kobe', 'mec')
SEEDS = (1, 2, 3)
ROUNDS = {'H8': (2, 3, 4), 'default': (2, 3)}
LINK = ('spike_positions.npy', 'channel_positions.npy', 'whitening_mat_inv.npy', 'provenance.json')


def folder(ex, setting, k, s):
    pre = '' if setting == 'H8' else 'default_'
    run = O / f'{ex}_{pre}repeat_s{s}'
    return run if k == ROUNDS[setting][-1] else run / f'round{k}'


def tonp(x):
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.asarray(x)


def weights(labels, clu, n_lab, n_unit):
    """Sparse n_lab x n_unit matrix: the share of each label's spikes in each unit."""
    C = sp.csr_matrix((np.ones(labels.size, np.float32), (labels, clu)), shape=(n_lab, n_unit))
    C.sum_duplicates()
    n = np.asarray(C.sum(1)).ravel()
    return sp.diags(1 / np.maximum(n, 1)) @ C


def roll_wall_only(wPCA, tF, Wall, spike_idx, clust_idx, dt):
    temps = torch.roll(Wall[clust_idx:clust_idx + 1] @ wPCA, shifts=dt, dims=2)
    if dt > 0:
        temps[:, :, :dt] = temps[:, :, dt].unsqueeze(-1)
    elif dt < 0:
        temps[:, :, dt:] = temps[:, :, dt - 1].unsqueeze(-1)
    Wall[clust_idx] = temps @ wPCA.T
    return tF, Wall


def merged(lab, t, Wall_lab, ops, device):
    """KS4's final merge of labels lab; returns labels, spike times and the spike order it left."""
    ops = dict(ops)
    ops['wPCA'] = torch.as_tensor(tonp(ops['wPCA'])).float().to(device)
    st = np.zeros((t.size, 3))
    st[:, 0] = t
    st[:, 2] = np.arange(t.size)
    tm.roll_features = roll_wall_only
    _, clu2, _, st2, _ = tm.merging_function(
        ops, torch.from_numpy(Wall_lab).float(), lab, st, torch.empty((t.size, 0, 0)),
        r_thresh=ops['settings']['merge_correlation_threshold_2'], device=device, check_dt=True)
    return clu2.astype(np.int64), st2[:, 0].astype(np.int64), st2[:, 2].astype(np.int64)


def build(src, dst, mode, device):
    dst.mkdir(parents=True, exist_ok=False)
    t = np.load(src / 'spike_times.npy').ravel().astype(np.int64)
    clu = np.load(src / 'spike_clusters.npy').ravel().astype(np.int64)
    tpl = np.load(src / 'spike_detection_templates.npy').ravel().astype(np.int64)
    W = np.load(src / 'templates.npy')
    nU = W.shape[0]
    order = None
    if mode == 'G':
        lab, tim = clu, t
        nL = nU
        for f in ('spike_times.npy', 'spike_clusters.npy', 'templates.npy'):
            os.link(src / f, dst / f)
    else:
        _, lab = np.unique(tpl, return_inverse=True)
        lab = lab.astype(np.int64)
        tim = t
        if mode == 'TM':
            P = weights(lab, clu, lab.max() + 1, nU)
            Wall = np.load(src / 'Wall.npy')
            Wall_lab = (P @ Wall.reshape(nU, -1).astype(np.float32)).reshape(-1, *Wall.shape[1:]).astype(np.float32)
            ops = np.load(src / 'ops.npy', allow_pickle=True).item()
            lab, tim, order = merged(lab, t, Wall_lab, ops, device)
            clu = clu[order]
        nL = int(lab.max()) + 1
        P = weights(lab, clu, nL, nU)
        Wl = (P @ W.reshape(nU, -1).astype(np.float32)).reshape(nL, *W.shape[1:]).astype(np.float32)
        np.save(dst / 'templates.npy', Wl)
        np.save(dst / 'spike_times.npy', tim)
        np.save(dst / 'spike_clusters.npy', lab.astype(np.int32))
    if order is None:
        for f in LINK:
            os.link(src / f, dst / f)
    else:
        for f in LINK:
            if f == 'spike_positions.npy':
                np.save(dst / f, np.load(src / f)[order])
            else:
                os.link(src / f, dst / f)
    return {'source': str(src), 'mode': mode, 'n_spikes': int(lab.size), 'n_labels': nL,
            'n_labels_with_spikes': int(np.unique(lab).size), 'n_units_source': int(np.unique(np.load(src / 'spike_clusters.npy')).size)}


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    plan = [(ex, g, k, s, m) for ex in EXS for g, rr in ROUNDS.items() for k in rr for s in SEEDS for m in ('T', 'TM')]
    plan += [(ex, 'H8', 3, 1, 'G') for ex in EXS]
    D.mkdir(parents=True, exist_ok=True)
    logf = D / 'variants.json'
    log = json.loads(logf.read_text()) if logf.exists() else {}
    for i, (ex, g, k, s, m) in enumerate(plan):
        name = f'{ex}_{g}_r{k}_s{s}_{m}'
        if name in log:
            continue
        tic = time.time()
        log[name] = build(folder(ex, g, k, s), D / name, m, device)
        logf.write_text(json.dumps(log, indent=1))
        print(f"[{i + 1}/{len(plan)}] {name}: {log[name]['n_labels_with_spikes']} labels, source {log[name]['n_units_source']} units; "
              f'{time.time() - tic:.0f} s', flush=True)
    print('done', flush=True)
