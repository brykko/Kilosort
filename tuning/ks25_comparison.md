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

**Population decoding** (spec §14): median error of tracked head direction (°) and 2D position (cm),
decoded from each sort's eligible units, cross-validated over the time halves, every sort of an excerpt
on the same windows. Three decoders: Bayesian (independent Poisson units, `bayes_decode.py`) and
population-vector correlation, with units weighted by rate ('raw') or z-scored ('z', `corr_decode.py`).
Lower is better; mean over replicates (SD across replicates 0.03–0.5); best at 20 ms in bold.

| Excerpt | Decoded | Decoder | Default 20 ms | KS2.5 20 ms | H8 20 ms | Default 100 ms | KS2.5 100 ms | H8 100 ms |
|---|---|---|---|---|---|---|---|---|
| Kobe | Direction | Bayesian | 12.3 | 12.6 | **11.4** | 9.0 | 9.7 | 9.1 |
| | | Correlation, raw | 13.8 | 14.9 | **11.6** | 8.3 | 9.0 | 7.6 |
| | | Correlation, z | **13.3** | 14.6 | 13.7 | 9.1 | 10.4 | 9.7 |
| Kobe | Position | Bayesian | 29.2 | 27.0 | **25.4** | 13.9 | 13.6 | 13.2 |
| | | Correlation, raw | 39.9 | 44.4 | **35.9** | 18.3 | 20.4 | 17.2 |
| | | Correlation, z | 28.2 | 28.0 | **26.7** | 15.3 | 15.1 | 15.3 |
| MEC | Direction | Bayesian | 18.0 | 18.0 | **16.3** | 11.9 | 12.3 | 11.5 |
| | | Correlation, raw | 20.3 | 22.0 | **18.6** | 11.4 | 12.3 | 11.9 |
| | | Correlation, z | 20.7 | 20.5 | **20.2** | 13.1 | 13.3 | 13.2 |
| MEC | Position | Bayesian | 26.9 | **21.9** | 23.1 | 13.0 | 12.0 | 12.7 |
| | | Correlation, raw | 38.3 | 40.6 | **37.0** | 16.2 | 17.1 | 18.1 |
| | | Correlation, z | 28.2 | 26.3 | **25.3** | 15.8 | 15.1 | 15.2 |

Decoding separates the sorters far less than S does. At 100 ms direction is saturated and position
nearly so, and which sorter leads depends on the decoder. At 20 ms, the least saturated window, H8
decodes best (or ties, MEC direction with z) in 10 of the 12 cases; default KS4 leads for kobe
direction with the z-scored correlation decoder, and KS2.5 for MEC position with the Bayesian decoder
only. With either correlation decoder H8 decodes MEC position best (see Population decoding below).

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

## Population decoding (both excerpts)

### Bayesian

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
- **Bayesian decoding and S rank MEC position differently.** S adds up each unit's own information; a
  Poisson decoder multiplies the units' likelihoods as if they were independent, so units that repeat
  others' information make it overconfident, and each weakly tuned unit's fitted map adds noise. The
  correlation decoders below, which don't multiply evidence across units, rank H8 first for MEC position
  at 20 ms.

### Correlation

Plan: [search_ks25.md](search_ks25.md) §7. Each window's spike-count vector is correlated (Pearson,
across units) with every visited bin's expected vector from the training maps, and the best-correlated
bin is the estimate; same units, maps, halves and windows as the Bayesian decoder. 'raw' uses counts and
maps as they are, so high-rate units weigh most; 'z' z-scores each unit first (counts over the half's
decoded windows, maps over the bins), so every unit weighs the same. Results: `runs\corr_decode.json` in
each data folder.

**Median error, all three windows** (mean over replicates; 20 / 100 / 500 ms):

| Excerpt | Decoded | Variant | KS4 default | KS2.5 standard | H8 |
|---|---|---|---|---|---|
| Kobe | Direction (°) | raw | 13.8 / 8.3 / 5.6 | 14.9 / 9.0 / 5.9 | 11.6 / 7.6 / 5.2 |
| | | z | 13.3 / 9.1 / 6.5 | 14.6 / 10.4 / 7.0 | 13.7 / 9.7 / 6.7 |
| Kobe | Position (cm) | raw | 39.9 / 18.3 / 12.6 | 44.4 / 20.4 / 13.2 | 35.9 / 17.2 / 12.5 |
| | | z | 28.2 / 15.3 / 12.2 | 28.0 / 15.1 / 12.0 | 26.7 / 15.3 / 12.4 |
| MEC | Direction (°) | raw | 20.3 / 11.4 / 8.6 | 22.0 / 12.3 / 9.4 | 18.6 / 11.9 / 9.5 |
| | | z | 20.7 / 13.1 / 10.6 | 20.5 / 13.3 / 10.1 | 20.2 / 13.2 / 10.3 |
| MEC | Position (cm) | raw | 38.3 / 16.2 / 12.8 | 40.6 / 17.1 / 13.3 | 37.0 / 18.1 / 15.0 |
| | | z | 28.2 / 15.8 / 13.2 | 26.3 / 15.1 / 12.6 | 25.3 / 15.2 / 12.9 |

**Equal unit counts** (100 ms; median error with N randomly drawn eligible units, mean over 10 draws and
the replicates):

| Excerpt | Decoded | Variant | Sorter | 25 | 50 | 100 | 200 | 400 | 800 | 1600 | all |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Kobe | Direction (°) | raw | default | 35.5 | 23.8 | 16.8 | 12.1 | 9.8 | | | 8.3 |
| | | | KS2.5 | 39.7 | 28.2 | 20.0 | 14.4 | 11.1 | 9.1 | | 9.0 |
| | | | H8 | 42.7 | 29.2 | 21.3 | 15.3 | 10.8 | 9.0 | 7.7 | 7.6 |
| | | z | default | 32.0 | 21.3 | 15.6 | 12.1 | 10.2 | | | 9.1 |
| | | | KS2.5 | 37.5 | 25.6 | 18.9 | 14.1 | 11.9 | 10.5 | | 10.4 |
| | | | H8 | 44.7 | 32.4 | 22.4 | 16.7 | 12.8 | 10.9 | 9.8 | 9.7 |
| Kobe | Position (cm) | raw | default | 65.1 | 54.0 | 40.2 | 27.3 | 21.3 | | | 18.3 |
| | | | KS2.5 | 66.7 | 57.4 | 47.9 | 34.8 | 25.3 | 20.9 | | 20.4 |
| | | | H8 | 69.0 | 60.7 | 48.7 | 36.2 | 25.3 | 20.0 | 17.4 | 17.2 |
| | | z | default | 57.7 | 44.4 | 29.8 | 20.5 | 16.7 | | | 15.3 |
| | | | KS2.5 | 61.3 | 48.4 | 34.4 | 22.8 | 17.5 | 15.4 | | 15.1 |
| | | | H8 | 64.3 | 55.7 | 41.4 | 27.8 | 20.5 | 17.0 | 15.5 | 15.3 |
| MEC | Direction (°) | raw | default | 44.7 | 32.0 | 23.1 | 16.4 | 13.3 | | | 11.4 |
| | | | KS2.5 | 46.9 | 38.2 | 27.4 | 19.3 | 15.6 | 12.7 | | 12.3 |
| | | | H8 | 51.0 | 40.9 | 30.1 | 25.0 | 17.1 | 14.2 | 12.2 | 11.9 |
| | | z | default | 41.3 | 30.1 | 22.1 | 17.0 | 14.4 | | | 13.1 |
| | | | KS2.5 | 45.9 | 33.5 | 24.1 | 18.4 | 15.4 | 13.5 | | 13.3 |
| | | | H8 | 53.4 | 43.7 | 31.2 | 23.8 | 18.2 | 15.2 | 13.5 | 13.2 |
| MEC | Position (cm) | raw | default | 57.5 | 49.8 | 40.6 | 28.3 | 20.6 | | | 16.2 |
| | | | KS2.5 | 57.6 | 51.0 | 41.5 | 32.9 | 23.5 | 17.8 | | 17.1 |
| | | | H8 | 61.9 | 55.9 | 49.1 | 40.7 | 28.1 | 22.1 | 18.8 | 18.1 |
| | | z | default | 54.6 | 45.8 | 35.2 | 23.9 | 18.2 | | | 15.8 |
| | | | KS2.5 | 56.6 | 47.9 | 36.5 | 26.0 | 18.8 | 15.3 | | 15.1 |
| | | | H8 | 60.2 | 54.7 | 45.3 | 33.5 | 22.9 | 17.8 | 15.5 | 15.2 |

- **Correlation decoding is less accurate than Bayesian** in most cases: at 100 ms position by 1.4–3.1 cm
  with z and 3.2–6.8 cm with raw, direction by 0.1–1.7° with z. The exception is direction with raw,
  0.7–1.5° better than Bayesian on kobe and within 0.5° on MEC.
- **At 20 ms H8 leads with both variants** except kobe direction with z (default 13.3° against H8's
  13.7°); for MEC position it leads with both (25.3 cm against 26.3 for KS2.5 with z; 37.0 against 38.3
  for default with raw), where the Bayesian decoder put KS2.5 first.
- **At longer windows the variants disagree:** raw, which lets high-rate units dominate, puts H8 last
  for MEC position at 100 and 500 ms (18.1 and 15.0 cm against 16.2 and 12.8 for default; H8 has ~445
  units above 5 Hz, against ~300 for default and ~140 for KS2.5), while z ranks the three within 0.7 cm.
- **At equal unit counts H8's units again decode worse**, with both variants, and its full population
  catches up through numbers, as with the Bayesian decoder.

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
- **Population decoding barely separates the three sorters.** Direction saturates at 100 ms windows
  and position nearly so, and the order there depends on the decoder. At 20 ms H8 decodes best or ties
  in 10 of 12 decoder and variable combinations, by 0.3–2.2° or 1.0–4.0 cm; KS2.5's lead for MEC
  position appears only with the Bayesian decoder. H8's large S gain shows up in decoding as a small
  improvement.

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
- Both: `bayes_decode.json` and `corr_decode.json` (Bayesian and correlation decoding of every sort;
  commands in [search_ks25.md](search_ks25.md) §6–7).
