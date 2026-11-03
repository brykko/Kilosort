"""Sort folders for scoring the set_cutoff-like cut: each scored KS4 sort (ks4_sort_scored.py) cut as in
apply_cut.py (strict, one pass, cut-off lowered from Th_learned + w to Th_learned), and a matched random
control: in every unit, the same number of spikes as the cut removed there, drawn at random (seeds 1-3).

H8 at widths 2 and 4, default at width 4; seed 1, both excerpts. A variant folder hard-links the sort's
files that don't depend on which spikes are kept (templates, channel maps, whitening, ops, params, KS4 labels)
and writes the per-spike arrays filtered; cleaning and evaluation outputs are left out, so the scoring
tools clean the variant afresh. Writes variants.json (spikes kept per variant) next to the folders.
"""
import json
import os
from pathlib import Path

import numpy as np

from set_cutoff import set_cutoff

FS = 30000
D = Path('C:/data/ks4_tuning/exploratory/autocurate')
PER_SPIKE = ('spike_times.npy', 'spike_clusters.npy', 'spike_templates.npy', 'spike_detection_templates.npy',
             'amplitudes.npy', 'spike_positions.npy', 'score.npy')
SKIP = ('cleaning.json', 'cleaning.npz', 'evaluation.json', 'comparison.json', 'full_st.npy', 'full_clu.npy',
        'full_amp.npy', 'kept_spikes.npy')
PLAN = {'H8': (2, 4), 'default': (4,)}
SEEDS = (1, 2, 3)


def write_variant(src, dst, keep):
    dst.mkdir(parents=True, exist_ok=False)
    for f in src.iterdir():
        if f.is_file() and f.name not in SKIP + PER_SPIKE:
            os.link(f, dst / f.name)
    for f in PER_SPIKE:
        np.save(dst / f, np.load(src / f)[keep])


def random_keep(clu, keep, rng):
    """Remove, in every unit, as many randomly chosen spikes as keep removes there."""
    out = np.ones(clu.size, bool)
    o = np.argsort(clu, kind='stable')
    _, start = np.unique(clu[o], return_index=True)
    for ix in np.split(o, start[1:]):
        n_cut = int(np.sum(~keep[ix]))
        if n_cut:
            out[rng.choice(ix, n_cut, replace=False)] = False
    return out


if __name__ == '__main__':
    log = {}
    for ex in ('kobe', 'mec'):
        for g, widths in PLAN.items():
            src = D / f'sorts/{ex}_{g}_s1'
            th = json.loads((src / 'provenance.json').read_text())['settings'].get('Th_learned', 8)
            t = np.load(src / 'spike_times.npy').ravel().astype(np.float64)
            clu = np.load(src / 'spike_clusters.npy').ravel().astype(np.int64)
            score = np.load(src / 'score.npy').astype(np.float64)
            for w in widths:
                keep, _ = set_cutoff(t, clu, score, FS, (th + w, th), strict=True)
                name = f'{ex}_{g}_cut{w}'
                write_variant(src, D / 'variants' / name, keep)
                log[name] = {'source': str(src), 'kept': int(keep.sum()), 'of': int(keep.size)}
                for s in SEEDS:
                    rk = random_keep(clu, keep, np.random.default_rng(s))
                    rname = f'{name}_rand{s}'
                    write_variant(src, D / 'variants' / rname, rk)
                    log[rname] = {'source': str(src), 'kept': int(rk.sum()), 'of': int(rk.size)}
                print(f'{name}: kept {keep.mean():.1%} of {keep.size} spikes (+ {len(SEEDS)} random controls)', flush=True)
    (D / 'variants' / 'variants.json').write_text(json.dumps(log, indent=2))
