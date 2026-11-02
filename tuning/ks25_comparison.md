# KS2.5 comparison: the winner against the lab's Kilosort 2.5 pipeline

Plan: [search_ks25.md](search_ks25.md); rules: [evaluation_spec.md](evaluation_spec.md) v0.17, §14.
KS2.5 is the neuropixels pipeline's Kilosort 2.5 (Gardner et al. 2022), preset "standard", from
`C:\code\github\neuropixels` at `6626f68` (clean), three replicates per excerpt (seeds 1–3 of
`learnAndSolve8b`; seed 1 is the pipeline's own sort). KS4 default and H8 are the sorts of
[round4_results.md](round4_results.md) and [transfer_results.md](transfer_results.md).

## Result

**H8 beats the lab's KS2.5 on both excerpts, by about 43% on each, with every guard passing on every
seed.** KS2.5 and default KS4 are close, and neither is ahead on both: KS2.5 is about δ worse than
default on kobe and 7 SD better on MEC.

| Excerpt | Score (bits/s) | KS4 default | KS2.5 standard | H8 | H8 − KS2.5 |
|---|---|---|---|---|---|
| Kobe | S (θ_ref) | 1019.7 ± 14.3 | 989.0 ± 5.9 | 1412.6 | **+423.6** |
| Kobe | S_track | 834.2 ± 13.6 | 816.9 ± 4.9 | 1175.2 | **+358.3** |
| MEC | Position S | 1955.0 ± 14.3 | 2054.0 ± 18.5 | 2932.3 | **+878.3** |

Means over replicates ± SD across them (default kobe: five seeds; the others three).

**Bayesian decoding** (spec §14, `bayes_decode.py`): median error of tracked head direction and 2D
position, decoded from each sort's eligible units, cross-validated over the time halves; every sort of
an excerpt is decoded on the same windows. Lower is better; mean over replicates (SD across replicates
0.03–0.3).

| Excerpt | Decoded (window) | KS4 default | KS2.5 standard | H8 | H8 − KS2.5 |
|---|---|---|---|---|---|
| Kobe | Direction (20 ms) | 12.3° | 12.6° | 11.4° | **−1.2°** |
| Kobe | Direction (100 ms) | 9.0° | 9.7° | 9.1° | −0.6° |
| Kobe | Position (20 ms) | 29.2 cm | 27.0 cm | 25.4 cm | **−1.6 cm** |
| Kobe | Position (100 ms) | 13.9 cm | 13.6 cm | 13.2 cm | −0.4 cm |
| MEC | Direction (20 ms) | 18.0° | 18.0° | 16.3° | **−1.7°** |
| MEC | Direction (100 ms) | 11.9° | 12.3° | 11.5° | −0.8° |
| MEC | Position (20 ms) | 26.9 cm | 21.9 cm | 23.1 cm | **+1.2 cm** |
| MEC | Position (100 ms) | 13.0 cm | 12.0 cm | 12.7 cm | +0.7 cm |

Decoding separates the sorters far less than S does. Direction is saturated at 100 ms (all three within
0.7°), and position nearly so; at 20 ms, the least saturated window, H8 decodes direction best on both
recordings and position best on kobe, but **KS2.5 decodes MEC position best**, although H8's position
S there is 878 bits/s higher (see Bayesian decoding below).

## Kobe (search excerpt, hybrid)

**KS2.5 against default** (`run_candidate.py --phase comparison`, runs 231–233 in the search log):

| Seed | S | ΔS | ΔS_track | Eligible | Hybrid agreement | Recovered | HD pairs merged | Untuned | Guards |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 990.2 | −29.4 | −15.1 | 899 | 0.978 | 30 | 0 | 333 | pass |
| 2 | 982.6 | −37.1 | −22.9 | 876 | 0.974 | 29 | 0 | 321 | recovered fails (29 < 31.4 − 1.8) |
| 3 | 994.2 | −25.5 | −13.9 | 892 | 0.976 | 30 | 0 | 333 | pass |
| Default | 1019.7 | | | ~772 | 0.951 | 31.4 | 0 | 193 | |

- KS2.5 is **worse than default by about δ**: mean ΔS −30.7 (δ = 28.6) and ΔS_track −17.3
  (δ_track/2 = 13.6). Its run-to-run spread (SD 5.9) is smaller than default's (14.3).
- **Hybrid units:** the spikes KS2.5 assigns to recovered units agree better with the truth (median
  0.976 against default's 0.951), but it recovers one or two fewer (29–30 against 30–32), and one
  replicate fails the recovered guard by one unit. No HD pair is merged.
- **Matched units** (§7.4, against default seed 1): about 405 of KS2.5's ~890 eligible units match a
  default unit, and score the same as their default counterparts (725–736 against 732–738 bits/s).
  KS2.5's other ~485 units carry ~260 bits/s against ~294 for default's ~373 unmatched units. So the
  shared cells are sorted equally well; KS2.5 loses on the units only one of the two finds.

**H8 against KS2.5** (`compare.py`, KS2.5's three replicates as the baseline, τ from their spread):

| H8 seed | ΔS | ΔS_track | Eligible (KS2.5 ~889) | Hybrid agreement (0.976, τ 0.003) | Recovered (29.7, τ 1.2) | Untuned (KS2.5 ~222) | Guards |
|---|---|---|---|---|---|---|---|
| 1 | +422.3 | +355.5 | 1712 | 0.976 | 33 | 658 | pass |
| 2 | +423.6 | +358.2 | 1689 | 0.975 | 33 | 631 | pass |
| 3 | +424.8 | +361.3 | 1683 | 0.977 | 33 | 613 | pass |

- **Matched units** (against KS2.5 seed 1): about 505 H8 units match a KS2.5 unit and score 885–895
  bits/s against 846–854 for their KS2.5 counterparts; H8's other ~1190 units carry ~523 bits/s
  against ~141 for KS2.5's ~395 unmatched units.

**Units reproduced across replicates** (`reproducibility.py`, all three pairs):

| Sorter | Eligible | Reproduced per pair | Reproduced | Fraction |
|---|---|---|---|---|
| KS4 default | ~770 | 662, 655, 648 | ~655 | 85% |
| KS2.5 standard | ~889 | 623, 632, 627 | ~627 | 70% |
| H8 | ~1695 | 1006, 995, 1012 | ~1004 | 59% |

KS2.5 finds ~120 more eligible units than default but reproduces ~30 fewer of them; H8 reproduces
~1.6 times as many units as KS2.5.

**Stricter duplicate cleaning** (spec §9.2, excess ≥ 1%): KS2.5 982.1, 974.6, 986.3 (mean 981.0, −8.0
from its S); default 995.7 (−24.0); H8 1389.9 (−22.7). KS2.5 is then −14.7 against default and H8
+408.9 against KS2.5. KS2.5 loses least to stricter cleaning: it leaves fewer near-threshold
duplicates (150–159 flagged pairs, 0.5–0.6% of spikes removed) and its sorts have no multi-unit
events (default 19–25, H8 76–80).

## MEC (transfer excerpt)

Tracked-position S (Appendix A), α = 0.0625 s (set on default's sorts again; the same value as the
transfer test). Time-shift control +4.4, +7.1, +6.7 bits/s, within 2 SD of 0.

**KS2.5 against default** (`transfer.py`; default 1955.0 ± 14.3, ~738 eligible, ~185 untuned):

| Seed | S | ΔS | Eligible | Untuned | Guards |
|---|---|---|---|---|---|
| 1 | 2066.9 | +112.0 | 887 | 310 | pass |
| 2 | 2062.2 | +107.2 | 870 | 293 | pass |
| 3 | 2032.8 | +77.8 | 890 | 308 | pass |

KS2.5 **beats default on MEC** by +99.0 (7 SD of default across seeds). H8, scored in the same run,
reproduces `transfer.json` exactly (2917.9, 2941.6, 2937.4).

**H8 against KS2.5** (`transfer.py --alpha 0.0625`, KS2.5's replicates as the baseline: S 2054.0 ±
18.5, ~882 eligible, ~221 untuned; time-shift control on KS2.5's sorts +11.2, +10.6, +11.8, within 2 SD
of 0):

| H8 seed | S | ΔS | Eligible | Untuned | Guards |
|---|---|---|---|---|---|
| 1 | 2917.9 | +863.9 | 1798 | 766 | pass |
| 2 | 2941.6 | +887.6 | 1862 | 819 | pass |
| 3 | 2937.4 | +883.4 | 1885 | 850 | pass |

Mean ΔS +878.3 (+43%, 48 SD of KS2.5 across its replicates).

**Units reproduced across replicates:** KS2.5 ~652 (645, 655, 656; 73–75% of ~882 eligible), against
~610 for default (82–84%) and ~900 for H8 (48–51%).

## Bayesian decoding (both excerpts)

Plan: [search_ks25.md](search_ks25.md) §6. Poisson decoder with independent units, flat prior and the
posterior maximum (Zhang et al. 1998); direction in 6° bins, position in 5 cm bins; tuning fitted on one
time half and the other decoded, both ways; only windows in which every tracking frame is usable
(valid tracking, speed ≥ 5 cm/s). Windows decoded: kobe 42,646 (20 ms), 8,324 (100 ms), 1,494 (500 ms);
MEC 49,737, 9,813 and 1,850. Results: `runs\bayes_decode.json` in each data folder.

**Median error, all three windows** (mean over replicates):

| Excerpt | Decoded | Window | KS4 default | KS2.5 standard | H8 |
|---|---|---|---|---|---|
| Kobe | Direction (°) | 20 / 100 / 500 ms | 12.3 / 9.0 / 5.8 | 12.6 / 9.7 / 6.2 | 11.4 / 9.1 / 5.8 |
| Kobe | Position (cm) | 20 / 100 / 500 ms | 29.2 / 13.9 / 10.8 | 27.0 / 13.6 / 10.8 | 25.4 / 13.2 / 10.8 |
| MEC | Direction (°) | 20 / 100 / 500 ms | 18.0 / 11.9 / 8.2 | 18.0 / 12.3 / 8.4 | 16.3 / 11.5 / 8.0 |
| MEC | Position (cm) | 20 / 100 / 500 ms | 26.9 / 13.0 / 10.9 | 21.9 / 12.0 / 10.0 | 23.1 / 12.7 / 10.8 |

**Saturation and equal unit counts** (100 ms windows; median error with N randomly drawn eligible
units, mean over 10 draws and the replicates; "all" is every eligible unit):

| Excerpt | Decoded | Sorter | 25 | 50 | 100 | 200 | 400 | 800 | 1600 | all |
|---|---|---|---|---|---|---|---|---|---|---|
| Kobe | Direction (°) | default | 25.3 | 17.3 | 12.9 | 10.9 | 9.5 | | | 9.0 |
| | | KS2.5 | 29.2 | 19.4 | 14.9 | 12.2 | 10.6 | 9.8 | | 9.7 |
| | | H8 | 36.0 | 22.7 | 16.0 | 12.2 | 10.7 | 9.5 | 9.1 | 9.1 |
| Kobe | Position (cm) | default | 58.7 | 41.9 | 26.5 | 18.3 | 15.3 | | | 13.9 |
| | | KS2.5 | 59.7 | 43.2 | 27.4 | 18.3 | 15.3 | 13.7 | | 13.6 |
| | | H8 | 65.5 | 51.1 | 33.3 | 21.2 | 16.3 | 14.2 | 13.3 | 13.2 |
| MEC | Direction (°) | default | 36.1 | 24.7 | 18.3 | 14.3 | 12.5 | | | 11.9 |
| | | KS2.5 | 36.9 | 27.6 | 19.8 | 15.5 | 13.5 | 12.4 | | 12.3 |
| | | H8 | 45.8 | 31.3 | 23.1 | 16.5 | 14.0 | 12.5 | 11.6 | 11.5 |
| MEC | Position (cm) | default | 52.2 | 41.4 | 30.2 | 18.9 | 14.6 | | | 13.0 |
| | | KS2.5 | 51.9 | 41.8 | 28.4 | 18.5 | 14.0 | 12.2 | | 12.0 |
| | | H8 | 56.2 | 49.3 | 37.1 | 25.4 | 17.6 | 14.3 | 12.9 | 12.7 |

- **Saturation:** every curve flattens towards its full count. Roughly doubling the units, from 400 to
  ~800 (default, KS2.5) or from 800 to ~1700 (H8), improves direction at 100 ms by 0.4–1.2° (4–10%) and
  position by 1.0–2.0 cm (7–15%): direction is close to saturated there, as expected, and position has a
  little more headway. The 20 ms window leaves the most.
- **At equal unit counts, H8's units decode worse** than default's or KS2.5's (400 units: kobe direction
  10.7° against 9.5° and 10.6°; MEC position 17.6 cm against 14.6 and 14.0). A random H8 unit carries
  less information than a random default unit, as S also shows: the units H8 adds over default carry
  about 0.4 bits/s each on kobe and 0.9 on MEC, against 1.3 and 2.6 for an average default unit. H8
  reaches the others' error through its larger number of units.
- **Decoding and S rank MEC position differently.** S adds up each unit's own information; a decoder's
  error depends on how the units' information combines, and on the noise each weakly tuned unit's fitted
  map adds. The two can rank sorts differently when the added units are weak or carry information the
  others already have. Which of these explains KS2.5's better MEC position decoding has not been tested.

## What the comparison shows

- **H8 is the better sorter on both recordings**, by ~420 bits/s of heading information on kobe and
  ~880 bits/s of position information on MEC (both +43% over KS2.5), without losing hybrid units,
  untuned units or eligible units against KS2.5. On the cells both sorters find, H8's versions carry
  ~5% more information; most of the gain comes from the ~1190 units KS2.5 doesn't have.
- **KS2.5 "standard" and default KS4 are close.** Both find similar numbers of units (KS2.5 ~120–150
  more), sort the cells they share equally well (kobe's matched units), and trade places between
  recordings: KS2.5 is about δ behind on kobe and ~7 SD ahead on MEC.
- **KS2.5's units reproduce less often than default KS4's** (69–75% against 82–85%) but more often than
  H8's (48–60%); in number, H8 reproduces ~1.4–1.6 times as many units as KS2.5.
- **Bayesian decoding barely separates the three sorters.** Direction saturates at 100 ms windows and
  position nearly so; at 20 ms H8 decodes best in three of the four cases (by 1.2–1.7° or 1.6 cm) and
  KS2.5 decodes MEC position best (by 1.2 cm). H8's large S gain shows up in decoding only as a small
  improvement, and not at all for MEC position.

## Sorter notes

- **Run time:** 23.5–27.9 min on kobe and 30.4–30.5 min on MEC (one MEC replicate took 42.0 min while
  harness jobs shared the CPU). Template learning stopped automatically after about 2,400–3,300 of
  the 16,000 learning batches; learning and spike extraction took ~17–23 min, preprocessing ~1.7,
  drift correction ~2 and autocurate ~3. KS4 default takes ~4–5 min and H8 ~7–9.
- **GPU:** the mex binaries (CUDA 12.9) run on the RTX 5070 Ti once MATLAB's CUDA forward compatibility
  is enabled (`ks25_sort.m` does it).
- **Information barrier:** the pipeline stores `curationTrackingTimeRange` in `rez.ops` and nothing reads
  it; the driver leaves it out.

## How it was run

Sorts (`<D>` the data folder of each excerpt; `<bin>` `search\hybrid\ap_1668-2868s_hybrid.bin` for kobe,
`search\ap_13575-14775s.bin` for MEC):

    python ks25_sort.py --data <D>/<bin> --meta <D>/search/source.ap.meta --probe <D>/search/probe_3a_opt3.json --out <D>/runs/ks25_standard_r<k> --seed <k>

Kobe scoring (`<K>` = `C:\data\ks4_tuning\26648_kobe_2020-08-21`):

    python run_candidate.py run --existing-sort <K>/runs/ks25_standard_r<k> --name ks25_standard --seed <k> --phase comparison --settings '{"sorter": "ks2.5 standard"}' --rationale "..."
    python compare.py --candidate <K>/runs/search/<H8 run>/sort --default <K>/runs/ks25_standard_r1 <K>/runs/ks25_standard_r2 <K>/runs/ks25_standard_r3 --raw-artefacts <K>/search/raw_artefacts_car.npz --out <K>/runs/ks25_comparison/H8_vs_ks25_s<k>.json
    python reproducibility.py --pair <K>/runs/ks25_standard_r1 <K>/runs/ks25_standard_r2 --pair ... --raw-artefacts <K>/search/raw_artefacts_car.npz --duration 1200 --hybrid-truth <K>/search/hybrid/hybrid_truth.npz --out <K>/runs/reproducibility_ks25.json
    python strict_cleaning.py --sorts <K>/runs/ks25_standard_r1 <K>/runs/ks25_standard_r2 <K>/runs/ks25_standard_r3 --out <K>/runs/strict_cleaning_ks25.json

(default's reproducibility the same way into `reproducibility_default.json`). MEC scoring (`<M>` =
`C:\data\ks4_tuning\25843_roger_2019-09-11`, tracking and raw-artefact arguments as in
[transfer_results.md](transfer_results.md)):

    python transfer.py --default <M>/runs/default_s1 ... s3 --winner <M>/runs/ks25_standard_r1 ... r3 <M>/runs/H8_s1 ... s3 ... --out <M>/runs/transfer_ks25.json
    python transfer.py --default <M>/runs/ks25_standard_r1 ... r3 --winner <M>/runs/H8_s1 ... s3 --alpha 0.0625 ... --out <M>/runs/transfer_H8_vs_ks25.json
    python reproducibility.py --pair <M>/runs/ks25_standard_r1 <M>/runs/ks25_standard_r2 --pair ... --raw-artefacts <M>/search/raw_artefacts_car.npz --duration 1200 --out <M>/runs/reproducibility_ks25.json

## Files

- Kobe (`C:\data\ks4_tuning\26648_kobe_2020-08-21\runs\`): `ks25_standard_r{1,2,3}\` (sorts, with
  `ks25_sort.log`, `provenance.json`, `evaluation.json`), `search\23{1,2,3}_ks25_standard_s*\`
  (comparison against default), `ks25_comparison\H8_vs_ks25_s{1,2,3}.json`,
  `reproducibility_ks25.json`, `reproducibility_default.json`, `strict_cleaning_ks25.json` (shadow
  folders under `strict_cleaning\`).
- MEC (`C:\data\ks4_tuning\25843_roger_2019-09-11\runs\`): `ks25_standard_r{1,2,3}\`,
  `transfer_ks25.json`, `transfer_H8_vs_ks25.json`, `reproducibility_ks25.json`.
- Both: `bayes_decode.json` (Bayesian decoding of every sort; command in
  [search_ks25.md](search_ks25.md) §6).
