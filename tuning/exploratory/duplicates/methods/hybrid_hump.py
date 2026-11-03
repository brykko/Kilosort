"""Do KS4 units that recover inserted (hybrid) units show the 0.27-0.5 ms within-unit hump, and which members are true?

Question: the inserted units have a fixed waveform (scaled copies) and known spike times. If KS4's
matcher double-detects a single clean spike at ~0.3-0.4 ms, the KS4 units matched to inserted units
should show the hump with one member a hit and the other a non-hit within the hump lags. If the hump
comes from the recording (doublet-shaped events, another neuron's spikes), hybrid units should show
little of it. For each inserted unit: its best-matching KS4 unit (most hits within +-0.4 ms); within
that unit, consecutive pairs at 8-15 vs 31-45 samples (per 1k spikes), split by member status
(hit = within +-3 samples of an inserted spike of that inserted unit). The truth trains' own lag
profile is reported too. Sorts: final outputs (after KS4's dedup) and full_st (before).
"""
from pathlib import Path

import numpy as np

TRUTH = Path(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\search\hybrid\hybrid_truth.npz')
AUTO = Path(r'C:\data\ks4_tuning\exploratory\autocurate\sorts')
HIT = 3


def pairs_by_status(t, hit, lo, hi):
    d = np.diff(t)
    j = np.where((d >= lo) & (d <= hi))[0]
    h = hit[j].astype(int) + hit[j + 1].astype(int)
    return np.bincount(h, minlength=3)            # 0 = neither, 1 = one hit, 2 = both


def main():
    z = np.load(TRUTH)
    tt, tu = z['spike_samples'], z['spike_unit']
    for name in ('kobe_H8_s1', 'kobe_default_s1'):
        fs = np.load(AUTO / name / 'full_st.npy')
        st_all = fs[:, 0].astype(np.int64)
        cl_all = np.load(AUTO / name / 'full_clu.npy').astype(np.int64)
        kept = np.load(AUTO / name / 'kept_spikes.npy')
        tot = {k: np.zeros(3, int) for k in ('kept_8_15', 'kept_31_45', 'before_1_6', 'truth_8_15', 'truth_31_45')}
        nsp_tot = 0; ntruth = 0
        for u in range(int(tu.max()) + 1):
            tr = np.sort(tt[tu == u])
            # best matching unit among kept spikes
            sk, ck = st_all[kept], cl_all[kept]
            lo_, hi_ = np.searchsorted(tr, sk - 12), np.searchsorted(tr, sk + 12)
            hitany = hi_ > lo_
            best = np.bincount(ck[hitany]).argmax()
            for label, mask in (('kept', kept), ('before', np.ones_like(kept))):
                m = mask & (cl_all == best)
                t = st_all[m]
                a, b = np.searchsorted(tr, t - HIT), np.searchsorted(tr, t + HIT)
                hit = b > a
                if label == 'kept':
                    tot['kept_8_15'] += pairs_by_status(t, hit, 8, 15)
                    tot['kept_31_45'] += pairs_by_status(t, hit, 31, 45)
                    nsp_tot += t.size
                else:
                    tot['before_1_6'] += pairs_by_status(t, hit, 1, 6)
            dd = np.diff(tr)
            tot['truth_8_15'][2] += np.sum((dd >= 8) & (dd <= 15))
            tot['truth_31_45'][2] += np.sum((dd >= 31) & (dd <= 45))
            ntruth += tr.size
        print(name, f'matched-unit spikes {nsp_tot}, truth spikes {ntruth}')
        for k, v in tot.items():
            n = ntruth if k.startswith('truth') else nsp_tot
            print(f'  {k:12s} pairs [neither, one hit, both hits] = {v.tolist()}  total per 1k spikes '
                  f'{1e3 * v.sum() / n:.3f}  (per lag-sample: {1e3 * v.sum() / n / (8 if "8_15" in k else 15 if "31_45" in k else 6):.3f})')


if __name__ == '__main__':
    main()
