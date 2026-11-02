# Transfer test results: MEC excerpt (spec §10)

Winner: H8 ([round4_results.md](round4_results.md)): `merge_bimodality_threshold` 0,
`cluster_resolution` 2.0, `merge_correlation_threshold_2` 0.95, `Th_universal` 7, `Th_learned` 7,
`cluster_downsampling` 10. Recording: rat 25843 ("roger"), 2019-09-11, `probe_1`, synced
13575–14775 s (spec §2). Scored with tracked-position S (Appendix A, `transfer.py`); default and
winner sorted on seeds 1–3 each, without hybrid insertion.

## Result

**The winner transfers.** Position S, default 1955.0 ± 14.3 bits/s (1944.1, 1949.7, 1971.1) against
H8 2917.9, 2941.6, 2937.4: **ΔS +977.3** (+50%, 68 SD of default across seeds). Both guards pass on
every seed:

| Seed | ΔS | Eligible (default ~738) | Untuned units (default ~185) |
|---|---|---|---|
| 1 | +962.9 | 1798 | 892 |
| 2 | +986.6 | 1862 | 965 |
| 3 | +982.4 | 1885 | 1000 |

- **Controls:** α = 0.0625 s, the best of the grid on default's held-out S without the null correction
  (0.25 s scores nearly the same; larger values lose). Time-shift control (tracking shifted 300 s):
  +4.4, +7.1, +6.7 bits/s, within 2 SD (28.5) of 0.
- **More units don't buy S by themselves:** default seed 1 with every eligible unit split at random
  in two scores +1.7 (1440 fragments); in four, −27.3 (2669 fragments).
- **Units reproduced across seeds** (`reproducibility.py`, all three seed pairs): ~900 for H8 (892,
  912, 896; 48–51% of its eligible units) against ~610 for default (610, 617, 603; 82–84%). H8 adds
  about 1110 eligible units, about 290 of them reproducible, at about 0.88 bits/s per added unit (33%
  of a default unit's mean S). Its reproduced fraction is lower than on kobe (49% against 59%), so on
  this recording it also makes more seed-dependent units.
- **Raw transients** (§6.1): 2 events reach ≥ 128 channels (kobe: none) and are masked (±1 ms each);
  1024 reach 64–127 channels.
- **Sort time:** 4–5 min for default, 6.6–8.9 min for H8.

For scale, the same score on kobe (default seeds 1–2 against H8 seed 1, hybrid units included, a
functional test of `transfer.py` only): ΔS +1137, so the winner's gain in position information is
of similar size on both recordings.

## How it was run

Extraction (`kilosort` environment for the excerpt; `npx-load` with the neuropixels `python` folder
on `PYTHONPATH` for tracking and probe):

    python extract_segment.py --remote-probe moser:sweeps/richarga/data/roger_25843/rec/2019-09-11_20-59-06/probe_1 --t0 13575 --t1 14775 --out-dir <M>/search
    python extract_tracking.py --motive-dir <M>/tracking_source/motive --t0 13575 --t1 14775 --out <M>/search/tracking_13575-14775s.npz
    python make_probe.py --meta-dir <folder with source.ap.meta> --n-chan-bin 385 --out <M>/search/probe_3a_opt3.json
    python raw_artefacts.py --data <M>/search/ap_13575-14775s.bin --probe <M>/search/probe_3a_opt3.json --n-chan-bin 385 --out <M>/search/raw_artefacts_car.npz

Sorts (seeds 1–3; `--settings` with the winner's settings for `H8_s<k>`, none for `default_s<k>`):

    python run_sort.py --data <M>/search/ap_13575-14775s.bin --probe <M>/search/probe_3a_opt3.json --n-chan-bin 385 --out <M>/runs/default_s<k> --seed <k>

Scoring:

    python transfer.py --default <M>/runs/default_s1 <M>/runs/default_s2 <M>/runs/default_s3 --winner <M>/runs/H8_s1 <M>/runs/H8_s2 <M>/runs/H8_s3 --tracking <M>/search/tracking_13575-14775s.npz --raw-artefacts <M>/search/raw_artefacts_car.npz --t0 13575 --duration 1200 --out <M>/runs/transfer.json

`<M>` is `C:\data\ks4_tuning\25843_roger_2019-09-11`.

## Files

`C:\data\ks4_tuning\25843_roger_2019-09-11\`: `search\` (excerpt with manifest, tracking, probe file,
raw-artefact pass), `runs\default_s{1,2,3}\`, `runs\H8_s{1,2,3}\` (sorts and their logs),
`runs\transfer.json` (scores, per-unit S_i, guards), `runs\reproducibility.json`.
