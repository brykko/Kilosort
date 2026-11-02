# Search round 3 results: detection and features

Plan: [search_round3.md](search_round3.md). Judged under spec v0.14. ΔS against the mean of the five
default replicates (S 1019.7 bits/s, 772 eligible units; δ = 28.6, δ/2 = 14.3). Base: R4 (bimodality
0, γ 2.0, first merge 0.975, final merge 0.95), mean ΔS +299.5.

## Leader

**C1: R4 + `Th_universal` 7 + `Th_learned` 7** (`merge_bimodality_threshold` 0,
`cluster_resolution` 2.0, `merge_correlation_threshold_1` 0.975, `merge_correlation_threshold_2`
0.95, `Th_universal` 7, `Th_learned` 7): ΔS +369.5, +374.5, +367.9 on seeds 1–3, **mean +370.6**
(ΔS_track +321.7), every guard passing on every seed; 32–34 of 40 hybrid units recovered, no HD pair
merged; about 1620 eligible units.

- **Stricter duplicate cleaning** (spec §9.2, `strict_cleaning.py`, excess ≥ 1%): S drops by 21.3 for
  C1 and 24.0 for default, so C1's advantage holds: **+373.3** over default re-scored the same way.
- **Units reproduced across seeds** (`reproducibility.py`, seeds 1–2): ~940, against ~960 for D5 and
  ~905 for R4. C1 adds about 94 eligible units over D5 and +27 bits/s, about 0.29 bits/s per added
  unit (22% of a typical default unit's S), unlike the 2–5% of the fragmenting γ steps; the lower
  `Th_learned` mostly adds low-amplitude spikes to existing cells. The difference in reproduced units
  is within the seed-to-seed spread (about 10 between R4's seed pairs).

## Screen (R4 + one change, seeds 1–3)

| Run | Change | ΔS seeds 1, 2, 3 | Mean ΔS | Mean ΔS_track | Eligible |
|---|---|---|---|---|---|
| **D5** | `Th_universal` 7 | +346.5, +346.0, +337.1 | **+343.2** | +287.5 | ~1525 |
| D2 | `Th_learned` 6 | +332.8, +319.1, +323.6 | +325.2 | +283.6 | ~1655 |
| D1 | `Th_learned` 7 | +317.8, +330.1, +326.2 | +324.7 | +274.4 | ~1515 |
| D4 | `Th_universal` 8 | +313.9, +324.3, +316.7 | +318.3 | +263.2 | ~1460 |
| D10 | `max_peels` 200 | +305.2, +301.0, +298.6 | +301.6 | +242.4 | ~1425 |
| D9 | `nblocks` 5 | +302.2, +305.1, +290.9 | +299.4 | +246.9 | ~1415 |
| D6 | `nearest_chans` 14 | +279.2, +271.6, +286.5 | +279.1 | +231.5 | ~1230 |
| D8 | `n_pcs` 9 | +245.5, +250.4, +243.8 | +246.6 | +202.0 | ~1225 |
| D3 | `Th_learned` 9 | +228.7, +239.2, +239.1 | +235.7 | +193.4 | ~1290 |
| D7 | `nearest_chans` 20 | +212.4, +234.1, +215.3 | +220.6 | +179.3 | ~1055 |

Every configuration passed every guard on every seed.

**Combination and extension:**

| Run | Configuration | ΔS seeds 1, 2, 3 | Mean ΔS | Mean ΔS_track | Eligible |
|---|---|---|---|---|---|
| **C1** | D5 + `Th_learned` 7 | +369.5, +374.5, +367.9 | **+370.6** | +321.7 | ~1620 |
| E1 | R4 + `Th_universal` 6 | +348.5, +347.4, +346.8 | +347.6 | +290.7 | ~1560 |

Units reproduced across seeds: D5 ~960 (63%), D1 ~880 (58%), C1 ~940 (58%).

## What round 3 shows

- **The first-pass detection threshold was the biggest lever:** `Th_universal` 7 gains +44 over R4
  and adds reproducible units (~960 against ~905): more cells get a template. It saturates at 7
  (6 gives the same).
- **The matching threshold adds information in existing units:** `Th_learned` 7 gains +25 alone and
  +27 on top of `Th_universal` 7; 6 gives the same as 7, 9 loses 64.
- **The peel limit doesn't matter for S** (`max_peels` 200: +2), although it binds in 89% of batches:
  the spikes it leaves behind carry no heading information.
- **Non-rigid drift correction makes no difference** (`nblocks` 5: −0.1).
- **Richer features hurt:** wider channel neighbourhoods (−20, −79) and more PCs (−53) give fewer
  units and lower S.

## Budget

36 search runs in round 3 (30 screen, 6 combination and extension); 172 runs in all, plus re-scores.
