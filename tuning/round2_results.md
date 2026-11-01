# Search round 2 results: early splitting, kept clean

Plan: [search_round2.md](search_round2.md). Judged under the current spec. ΔS against the mean of
the five default replicates (S 1019.7 bits/s, 772 eligible units; δ = 28.6, δ/2 = 14.3);
ΔS_track against their S_track (834.2; δ_track/2 = 13.6). Default recovers 31.4 of 40 hybrid units
and removes 1.18% of spikes in cleaning.

## Leader

**R4: γ 2.0 with bimodality 0, first merge 0.975, final merge 0.95**
(`merge_bimodality_threshold` 0, `cluster_resolution` 2.0, `merge_correlation_threshold_1` 0.975,
`merge_correlation_threshold_2` 0.95): ΔS +293.1, +298.4, +307.1 on seeds 1–3, **mean +299.5**
(ΔS_track +246.3, 29% of S), every guard passing on every seed (32 of 40 hybrid units recovered,
no HD pair merged; cleaning removes 1.04–1.14% of spikes).
About 1420 eligible units against default's 772.

Where the gain sits (matched-unit report, seed 1): the 577 units that match a default unit score the
same as their default counterparts (907 vs 905 bits/s); R4's 830 other units carry 405 bits/s
against 125 for default's 200 unmatched units. R4 replaces poorly separated default clusters with
more, smaller units that together carry about three times the information.

## Three-seed results

| Run | Configuration | ΔS seeds 1, 2, 3 | Mean ΔS | Mean ΔS_track | Eligible | Guard failures |
|---|---|---|---|---|---|---|
| **R4** | **γ 2.0 + first merge 0.975 + final merge 0.95** | +293.1, +298.4, +307.1 | **+299.5** | +246.3 | ~1420 | none |
| F2 | R4 with final merge 0.99 | +293.5, +298.9, +308.1 | +300.2 | +246.8 | ~1440 | none |
| R16 | R4 + bottom-up tree walk | +298.9, +296.7, +300.8 | +298.8 | +245.2 | ~1700 | none |
| R7 | γ 1.0 + first merge 0.975 + final merge 0.95 | +278.6, +269.4, +278.7 | +275.6 | +225.4 | ~1260 | none |
| R2 | `standard` + final merge 0.95 | +260.7, +261.6, +265.1 | +262.4 | +212.8 | ~1145 | none |
| R1 | `standard` + final merge 0.85 | +253.1, +255.8, +259.1 | +256.0 | +208.7 | ~1080 | none |
| R5 | `standard` + γ 2.0 | +202.3, +227.8, +228.4 | +219.5 | +185.7 | ~1300 | none |
| R12 | `standard` + downsampling 10 | +211.1, +208.6, +220.4 | +213.4 | +180.5 | ~1140 | none |
| R9 | tree connectivity 1.45 + first merge 0.975 + final merge 0.95 | +208.8, +211.9, +207.4 | +209.4 | +168.7 | ~1030 | none |
| R10 | `standard` + neighbours 5 | +208.3, +196.7, +189.9 | +198.3 | +166.4 | ~1470 | none |
| R8 | `standard` + γ 1.0 | +193.1, +193.7, +203.3 | +196.7 | +165.8 | ~1145 | none |
| R14 | `standard` + bottom-up tree walk | +175.5, +174.8, +195.5 | +182.0 | +152.8 | ~1045 | none |
| R3 | γ 2.0 + first merge 0.975 | +162.0, +194.8, +183.6 | +180.2 | +152.0 | ~1245 | none |
| R6 | γ 1.0 + first merge 0.975 | +157.2, +172.4, +165.0 | +164.9 | +139.8 | ~1115 | none |
| R11 | `standard` + neighbours 20 | +123.4, +112.6, +114.1 | +116.7 | +98.5 | ~860 | none |

For reference from round 1: `standard` alone +178.6; γ 2.0 / 1.0 with bimodality 0 alone +181.8 /
+155.7.

R13 and R15 (refractory significance, small-group floor) were dropped by the preparation diagnostic
(no effect on this data). Run 132 (R4 + neighbours 5) was stopped: its 7425 first-pass clusters needed
~26 GB of GPU memory in template matching and spilled to system memory. Run 101 (R1, seed 3) was
stopped during the two-job attempt and re-run as 118.

## Resolution sweep (seed 1, diagnostic)

| γ (R4 otherwise) | ΔS | Eligible | Spikes removed in cleaning | Flagged duplicate pairs per unit |
|---|---|---|---|---|
| 2.0 (R4) | +293.1 | 1407 | 1.14% | 0.351 |
| 2.65 | +300.5 | 1523 | 1.18% | 0.348 |
| 4 | +314.0 | 1807 | 1.19% | 0.389 |

S still rises at γ 4, but slowly: about 280 more eligible units for +13.5 bits/s. γ 6–8 and neighbours 3 were not run: their first-pass cluster counts (estimated
7000–8500) put template matching beyond GPU memory.

**Seed 2 and the stopping rule.** γ 4: ΔS +316.6 (seeds 1–2 mean +315.3); γ 8 (now within GPU memory
after the sparse template cross-correlations): +327.3 and +332.8 (mean +330.1), all guards passing.
Units that reproduce across seeds (`reproducibility.py`, seeds 1 and 2):

| Configuration | Eligible units | Reproduced | Units reproduced |
|---|---|---|---|
| Default | ~770 | 85% | ~660 |
| Final merge 0.7 | ~795 | 87% | ~690 |
| `standard` | ~1030 | 79% | ~810 |
| **R4 (γ 2)** | ~1410 | 64% | **~905** |
| γ 4 | ~1820 | 48% | ~870 |
| γ 8 | ~2460 | 33% | ~815 |

The number of reproduced units peaks at γ 2 and falls beyond it, while S creeps up (+15 per doubling
of γ, about δ/2) on units that each carry 2–5% of a typical unit's S: the extra units are
seed-dependent fragments. By the spec's resolution stopping rule (§9.2), the operating point is R4.

## Over-splitting check (R4, seed 1; `oversplit.py`)

Every eligible unit split into k fragments, eligibility re-applied, S scored (baseline 1312.7):

| k | Random | By amplitude | By time block |
|---|---|---|---|
| 2 | −3.6 | +40.5 | −1312.7 |
| 4 | −29.3 | +31.3 | −1312.7 |
| 8 | −158.0 | −101.7 | −1312.7 |
| 16 | −578.0 | −534.7 | −1312.7 |

S has an optimum along the resolution axis. Random two-way splits are free, as the per-unit
estimation cost is removed by the null correction; beyond that, fragments lose to estimation noise
and to the 100-spikes-per-half floor. Time-block fragments never meet the floor, so drift-like
splits remove units from S entirely. Amplitude splits gain at k = 2–4, which could mean remaining
mixtures, but amplitude also carries heading on its own (spikes within bursts are smaller, and burst
rate rises near the preferred direction), so this is not evidence of mixtures by itself.

## What round 2 shows

- **The effects add.** Graph resolution with the tree's bimodality merges off (γ), a stricter first
  merge and a stricter final merge each contribute: the final merge at 0.95 adds +84 to +119 to
  every base; γ 2.0 adds +15 to +24 over γ 1.0.
- **The stricter first merge reduces duplicates.** With first merge 0.975, γ 2.0 with bimodality 0
  needs less duplicate cleaning (1.22–1.27% of spikes removed, against 1.29–1.41% without it).
- **The final merge saturates at 0.95:** 0.99 gives the same S.
- **γ is flattening by 4** (single seed).
- **The bottom-up tree walk doesn't change S** but needs less duplicate cleaning (R16: 0.85–0.93%
  of spikes removed, against R4's 1.04–1.14%).
- **Graph settings with the tree relaxed:** fewer neighbours raises S (5: +20 over `standard`; 20:
  −62) but multiplies first-pass clusters, which makes it impractical together with γ. Downsampling
  10 adds +35.
- **Tree connectivity is a weaker route** than γ (+209 with both merges, against +299).

## Budget

48 search runs in round 2 (42 screen and R16 runs, 6 refinement, 2 sweep; plus runs 101 and 132,
stopped), 84 in all so far.
