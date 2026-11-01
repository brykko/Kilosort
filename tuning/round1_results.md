# Search round 1 results: clustering resolution

Plan: [search_round1.md](search_round1.md). Spec v0.7 for runs 001–023, v0.8 (tracked-heading
condition) from run 024. ΔS against the mean of the five default replicates (S 576.2 bits/s;
δ = 16.0); ΔS_track against their S_track (483.9; δ_track/2 = 7.7). Default: 364.6 eligible
units, 2.8 suspected splits (G3 limit 4.99), 31.4 of 40 hybrid units recovered.

## Leader

**`merge_correlation_threshold_2` = 0.7** (final merge; default 0.5).

| Seed | ΔS | ΔS_track | Eligible | Units | G3 splits | Hybrid recovered | Guards |
|---|---|---|---|---|---|---|---|
| 1 | +39.3 | +32.0 | 379 | 842 | 3 | 30 | all pass |
| 2 | +44.1 | +36.2 | 385 | 831 | 4 | 32 | all pass |
| 3 | +34.4 | +27.9 | 382 | 827 | 3 | 32 | all pass |
| **Mean** | **+39.3** (2.5δ, 6.8% of S) | **+32.0** | 382 | 833 | | | |

## All runs

| Run | Settings | ΔS | ΔS_track | Eligible | Units | G3 | Hybrid | Guard failures |
|---|---|---|---|---|---|---|---|---|
| 001 S1 | `standard` preset | +66.3 | +56.2 | 460 | 1081 | 16 | 31 | G3, G5, G6 pairs, G9 |
| 002 S2 | shape criterion | +33.7 | +29.5 | 382 | 845 | 7 | 30 | G3 |
| 003 S3 | bimodality 0.4 | +9.9 | +7.5 | 395 | 920 | 3 | 31 | none |
| 004 S3 | bimodality 0.2 | +29.8 | +29.9 | 417 | 989 | 9 | 31 | G3, G5, G6 pairs |
| 005 S3 | bimodality 0 | +25.3 | +22.8 | 419 | 1022 | 4 | 32 | G5, G6 both |
| 006 S4 | tree connectivity 1.0 | +3.8 | +2.1 | 384 | 907 | 2 | 32 | G5 |
| 007 S4 | tree connectivity 1.45 | +18.4 | +18.7 | 393 | 931 | 5 | 31 | G3, G5 |
| 008 S5 | first merge 0.95 | +11.7 | +11.6 | 369 | 798 | 3 | 29 | G4 recovered |
| 009 S5 | first merge 0.975 | −15.4 | −12.7 | 361 | 804 | 3 | 31 | none |
| 010 S6 | **final merge 0.7** | +39.3 | +32.0 | 379 | 842 | 3 | 30 | none |
| 011 S6 | final merge 0.85 | +74.3 | +59.1 | 402 | 870 | 11 | 30 | G3, G9 |
| 012 S7 | `ccg_threshold` 0.15 | +13.9 | +11.0 | 369 | 825 | 3 | 30 | none |
| 013 S8 | `acg_threshold` 0.1 | −1.2 | −1.9 | 361 | 816 | 3 | 30 | none |
| 014 S9 | γ 1.0, bimodality 0 | +14.2 | +12.3 | 431 | 1127 | 4 | 32 | G5, G6 pairs |
| 015 S9 | γ 1.35, bimodality 0 | +0.9 | +0.4 | 427 | 1187 | 6 | 31 | G3, G5, G6 both |
| 016 S9 | γ 2.0, bimodality 0 | +40.7 | +38.2 | 479 | 1305 | 8 | 31 | G3, G5, G6 pairs |
| 017 S10 | neighbours 5 | −6.2 | −4.1 | 367 | 852 | 1 | 31 | G5, G6 pairs |
| 018 S10 | neighbours 20 | +10.6 | +9.6 | 351 | 740 | 5 | 31 | G3, G7 |
| 019 S11 | downsampling 10 | −12.8 | −13.6 | 377 | 880 | 2 | 31 | G5, G6 pairs |
| 020 S11 | downsampling 10, subset 50k | −12.8 | −13.6 | 377 | 880 | 2 | 31 | G5, G6 pairs (identical to 019: no centre reaches the subset cap) |
| 021, 022 | final merge 0.7, seeds 2, 3 | +44.1, +34.4 | +36.2, +27.9 | 385, 382 | | 4, 3 | 32, 32 | none |
| 023 | `ccg_threshold` 0.10 (range extension) | +21.4 | +16.4 | 378 | 836 | 5 | 30 | G3 |
| 024–026 | C1: final merge 0.7 + `ccg_threshold` 0.15, seeds 1–3 | +48.8, +56.7, +44.2 (mean +49.9) | +40.1, +45.6, +35.6 | 383–393 | | 4, 5, 4 | 30–32 | G3 on seed 2 (5 vs 4.99): not adopted |
| 027 | C2: final merge 0.7 + bimodality 0.4 | +47.8 | +38.0 | 413 | 953 | 3 | 31 | G6 both: dropped |

No run merged a hybrid HD pair. S and S_track agree throughout (r = 0.995, ΔS_track ≈ 0.85 ΔS).

## What round 1 shows

- **Fewer merges at the final merge step is the main lever.** Default KS4 merges pairs whose
  templates correlate above 0.5 and whose CCG passes its refractoriness test. Raising the threshold
  to 0.7 gains 39 bits/s cleanly; 0.85 gains 74 but leaves 11 same-neuron pairs unmerged (G3). So
  among default's final merges, many join differently tuned cells, and some join halves of one cell.
  The shape criterion (S2) and `ccg_threshold` (S7) act on the same merges and point the same way.
- **More units is not the same as more resolution.** Settings that split earlier in the pipeline
  (bimodality 0, γ, tree connectivity) add 30–170 eligible units but gain less per unit, and they
  fail G5 and G6: many of the extra units are small, near-noise, or duplicates of others.
- **The graph clustering's own settings don't help.** γ is erratic (+14, +1, +41), and neighbours,
  downsampling and the subset cap gain nothing.
- **The first merge should stay where it is:** 0.95 loses a hybrid unit, 0.975 costs 15 bits/s.

## Budget

27 of the 85 search runs used (17 of round 1's 44 unused); 15 reserved runs untouched. About 2.3 hours
of GPU.
