"""Are the KS4 spikes that KS2.5 doesn't output the small ones? Seed 1, labels as trace_units.py.

Within each KS4 eligible unit with >= 50 spikes both in and not in KS2.5's output: KS4's per-spike
amplitude (amplitudes.npy) of the spikes not in KS2.5's output relative to the median of those in it, and
the share of them below the 5th percentile of those in it (5% if they were alike). Also the spikes of
the unit's refractory violations that KS2.5 doesn't output. Between units: KS4's template amplitude
(cluster_Amplitude.tsv) of shared units, own units drawing >= 10% of their spikes from a KS2.5 unit, and
own units that don't.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import trace_units as tu  # noqa: E402
from cleaning import load_clean_units  # noqa: E402


def wmedian(x, w):
    o = np.argsort(x)
    cw = np.cumsum(np.asarray(w, float)[o])
    return np.asarray(x)[o][np.searchsorted(cw, cw[-1] / 2)]


def per_spike_amps(sort, ex, units):
    base = tu.K if ex == 'kobe' else tu.M
    by, _, _, amp = load_clean_units(sort, base / 'search/raw_artefacts_car.npz', np.load(sort / 'amplitudes.npy').ravel())
    return {c: amp[c][np.argsort(by[c], kind='stable')] for c in units}


def template_amps(sort):
    rows = [r.split('\t') for r in (sort / 'cluster_Amplitude.tsv').read_text().splitlines()[1:] if r.strip()]
    return {int(c): float(a) for c, a in rows}


for ex, s in tu.SORTS.items():
    by25, d25, T25, el25 = tu.load(s['ks25'], ex)
    for g in ('default', 'H8'):
        by4, d4, T4, el4 = tu.load(s[g], ex)
        pr = tu.pairs({c: by4[c] for c in el4}, d4, {c: by25[c] for c in el25}, d25)
        sh4 = {a for a, _ in pr}
        lab = tu.label({c: by4[c] for c in el4}, d4, by25, d25, (T4 + T25) / 2)
        amp = per_spike_amps(s[g], ex, el4)
        tamp = template_amps(s[g])
        print(f'\n=== {ex}: {g} against KS2.5 ===', flush=True)
        for grp in ('shared', 'own'):
            units = [c for c in el4 if (c in sh4) == (grp == 'shared')]
            ratio, w, below, n_out, vratio, vw, n_all_out = [], [], 0, 0, [], [], 0
            for c in units:
                a, l = amp[c], lab[c]
                n_all_out += np.sum(l < 0)
                inn, out = a[l >= 0], a[l < 0]
                if inn.size < 50 or out.size < 50:
                    continue
                med = np.median(inn)
                ratio.append(np.median(out) / med); w.append(out.size)
                below += np.sum(out < np.percentile(inn, 5)); n_out += out.size
                i, j = tu.violation_pairs(by4[c])
                culprit = np.unique(np.r_[i[l[i] < 0], j[l[j] < 0]])
                if culprit.size:
                    vratio.append(np.median(a[culprit]) / med); vw.append(culprit.size)
            print(f'  {grp} units: {len(ratio)} of {len(units)} units, holding {n_out / max(n_all_out, 1):.0%} of the spikes KS2.5 doesn\'t output, '
                  f'have both kinds: those spikes\' amplitude / median of the unit\'s spikes in KS2.5 = {wmedian(ratio, w):.2f} (spike-weighted median); '
                  f'{below / n_out:.0%} below the 5th percentile of those in KS2.5; violation spikes not in KS2.5: {wmedian(vratio, vw):.2f}')
        cls = {'shared': [], 'own, from a KS2.5 unit': [], 'own, not from any KS2.5 unit': []}
        for c in el4:
            if c in sh4:
                cls['shared'].append(c)
            else:
                k = tu.contributors(lab[c], by4[c].size)[0]
                cls['own, from a KS2.5 unit' if k else 'own, not from any KS2.5 unit'].append(c)
        print('  template amplitude (cluster_Amplitude.tsv), median [quartiles]: ' + '; '.join(
            f'{k} ({len(v)}): {np.median([tamp[c] for c in v]):.1f} [{np.percentile([tamp[c] for c in v], 25):.1f}, '
            f'{np.percentile([tamp[c] for c in v], 75):.1f}]' for k, v in cls.items() if v), flush=True)
