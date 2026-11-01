# Search round 1 results: clustering resolution

Plan: [search_round1.md](search_round1.md). Judged under the current spec ([evaluation_spec.md](evaluation_spec.md)):
every sort was scored with the current eligibility and guards (`--existing-sort` re-scores, not
counted in the budget). ΔS against the mean of the five default replicates (S 1019.7 bits/s, 772
eligible units; δ = 28.6, δ/2 = 14.3); ΔS_track against their S_track (834.2; δ_track/2 = 13.6).
Default recovers 31.4 of 40 hybrid units.

## Leader

**The `standard` preset** (bimodality 0, shape criterion, first merge 0.975): ΔS +172.7, +176.5,
+186.5 on seeds 1–3, **mean +178.6** (ΔS_track +149.2), every guard passing on every seed.

## Three-seed results

| Configuration | ΔS seeds 1, 2, 3 | Mean ΔS | Mean ΔS_track | Eligible | Guard failures |
|---|---|---|---|---|---|
| Final merge 0.7 | +63.2, +59.1, +44.1 | +55.5 | +43.5 | ~795 | none |
| Final merge 0.7 + `ccg_threshold` 0.15 | +71.4, +72.2, +55.9 | +66.5 | +52.1 | ~800 | none |
| **`standard` preset** | +172.7, +176.5, +186.5 | **+178.6** | +149.2 | ~1030 | none |
| γ 2.0, bimodality 0 | +184.3, +189.9, +171.3 | +181.8 | +155.0 | ~1225 | seed 2: spikes removed 1.41% (limit 1.34%), duplicate pairs 0.45 per unit (limit 0.41) |
| γ 1.0, bimodality 0 | +149.2, +156.2, +161.7 | +155.7 | +131.9 | ~1095 | seeds 2, 3: spikes removed 1.38–1.41% |

## Single runs (seed 1)

| Settings | ΔS | ΔS_track | Eligible | Hybrid recovered | Guard failures |
|---|---|---|---|---|---|
| Shape criterion | +42.3 | +33.2 | 796 | 30 | none |
| Bimodality 0.4 | +85.1 | +73.7 | 875 | 31 | none |
| Bimodality 0.2 | +114.3 | +99.7 | 940 | 31 | none |
| Bimodality 0 | +136.0 | +114.3 | 979 | 32 | spikes removed |
| Tree connectivity 1.0 | +87.2 | +74.3 | 865 | 32 | none |
| Tree connectivity 1.45 | +93.0 | +76.7 | 884 | 31 | none |
| Tree connectivity 1.9 | +102.6 | +87.6 | 899 | 31 | hybrid agreement |
| First merge 0.95 | −8.7 | −6.4 | 762 | 29 | hybrid recovered |
| First merge 0.975 | −3.4 | −5.2 | 767 | 31 | none |
| Final merge 0.85 | +102.5 | +80.1 | 826 | 30 | none |
| Final merge 0.95 | +109.5 | +85.5 | 846 | 30 | none |
| `ccg_threshold` 0.15 | +25.4 | +19.5 | 788 | 30 | none |
| `ccg_threshold` 0.10 | +42.2 | +32.7 | 799 | 30 | none |
| `acg_threshold` 0.1 | +8.9 | +7.1 | 779 | 30 | none |
| γ 1.35, bimodality 0 | +159.7 | +134.2 | 1123 | 31 | spikes removed, duplicate pairs |
| γ 2.65, bimodality 0 | +190.3 | +160.6 | 1337 | 32 | spikes removed, duplicate pairs |
| Neighbours 5 | −10.6 | −5.1 | 787 | 31 | duplicate pairs |
| Neighbours 20 | −9.4 | −10.3 | 719 | 31 | untuned units |
| Downsampling 10 (with or without a 50k subset cap: identical, no centre reaches the cap) | +16.1 | +13.5 | 821 | 31 | duplicate pairs |
| Final merge 0.7 + bimodality 0.4 | +146.3 | +120.3 | 901 | 31 | spikes removed |

No run merged a hybrid head-direction pair. S and S_track move together (ΔS_track ≈ 0.84 ΔS).

## What round 1 shows

- **Splitting earlier in the pipeline gives the largest gains.** Relaxing the merge tree
  (bimodality 0, the `standard` preset) or raising the graph resolution (γ with bimodality 0) gains
  +136 to +190 bits/s, about three times what the final-merge settings gain (+55 to +110). Tracked
  heading follows.
- **Some of them sit at the cleaning limits.** γ with bimodality 0 produces more duplicate spikes
  (removed before scoring) and fails the cleaning guards on one or two seeds, by small margins; the
  `standard` preset stays inside them.
- **Final-merge settings gain cleanly.** Raising `merge_correlation_threshold_2` (0.7, 0.85, 0.95)
  and lowering `ccg_threshold` pass every guard; the shape criterion acts on the same merges.
- **The first merge should stay where it is:** 0.95 loses a hybrid unit, 0.975 costs a little.
- **Neighbours and downsampling were screened only at default tree settings**, where the tree
  re-merges most of what the clustering produces; they stay in round 2, tested together with merge
  settings.

## Budget

36 of the 85 search runs used (round 1 had 44); the 15 reserved runs are untouched.
