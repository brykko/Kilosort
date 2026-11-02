# Search round 4 results: preprocessing, universal templates, resolution recheck

Plan: [search_round4.md](search_round4.md). Judged under spec v0.14. ΔS against the mean of the five
default replicates (S 1019.7 bits/s, 772 eligible units; δ = 28.6, δ/2 = 14.3). Base: W, round 3's
leader after ablation ([round3_results.md](round3_results.md), Ablation): bimodality 0, γ 2.0,
final merge 0.95, `Th_universal` 7, `Th_learned` 7; mean ΔS +368.6.

## Leader

**H8: W + `cluster_downsampling` 10** (`merge_bimodality_threshold` 0, `cluster_resolution` 2.0,
`merge_correlation_threshold_2` 0.95, `Th_universal` 7, `Th_learned` 7, `cluster_downsampling` 10):
ΔS +391.6, +393.0, +394.2 on seeds 1–3, **mean +392.9** (ΔS_track +341.0), every guard passing on
every seed; 33 of 40 hybrid units recovered on each seed, no HD pair merged; about 1695 eligible
units.

- **Units reproduced across seeds** (`reproducibility.py`, all three seed pairs): ~1004 (1006, 995,
  1012), against ~941 for W (951, 926, 947). The ~60 extra reproducible units are well beyond W's
  pair-to-pair spread (25). H8 adds about 116 eligible units for +24.3 bits/s, about 0.21 bits/s per
  added unit (16% of a typical default unit's S).
- **Stricter duplicate cleaning** (spec §9.2): S drops by 22.7 for H8 and 24.0 for default; H8 stays
  **+394.2** over default re-scored the same way.
- **Matched units** (§7.4, seed 1): the 520 units that match a default unit score 898 bits/s against
  875 for their default counterparts; H8's other 1192 units carry 513 bits/s against 155 for
  default's 257 unmatched units.

## Screen (W + one change, seeds 1–3)

| Run | Change | ΔS seeds 1, 2, 3 | Mean ΔS | vs W | Mean ΔS_track | Eligible |
|---|---|---|---|---|---|---|
| **H8** | `cluster_downsampling` 10 | +391.6, +393.0, +394.2 | **+392.9** | **+24.3** | +341.0 | ~1695 |
| G2 | `cluster_resolution` 3.0 | +383.0, +390.6, +378.9 | +384.2 | +15.6 | +332.8 | ~1830 |
| H5 | `dminx` 16 | +373.8, +387.7, +384.5 | +382.0 | +13.4 | +332.4 | ~1630 |
| H7 | `n_templates` 9 | +367.0, +378.8, +365.6 | +370.5 | +1.9 | +320.9 | ~1580 |
| H1 | `highpass_cutoff` 150 | +373.8, +368.5, +364.1 | +368.8 | +0.2 | +315.7 | ~1555 |
| H6 | `template_sizes` 8 | +376.8, +356.6, +367.5 | +367.0 | −1.6 | +317.9 | ~1560 |
| H4 | `whitening_range` 64 | +365.0, +359.7, +363.2 | +362.7 | −5.9 | +313.9 | ~1580 |
| G1 | `cluster_resolution` 1.5 | +360.6, +360.8, +355.6 | +359.0 | −9.6 | +309.8 | ~1470 |
| H3 | `whitening_range` 16 | +359.1, +361.3, +352.8 | +357.7 | −10.9 | +308.5 | ~1570 |
| H2 | `highpass_cutoff` 500 | +325.5, +335.8, +334.6 | +331.9 | −36.7 | +288.1 | ~1580 |

Every configuration passed every guard on every seed.

**Range extension:**

| Run | Configuration | ΔS seeds 1, 2, 3 | Mean ΔS | vs H8 | Mean ΔS_track | Eligible |
|---|---|---|---|---|---|---|
| X1 | W + `cluster_downsampling` 5 | +400.2, +398.3, +400.6 | +399.7 | +6.8 | +346.2 | ~1830 |

## Resolution (spec §9.2)

Units reproduced across seeds (mean over the three seed pairs, except X1: seeds 1–2):

| Configuration | Eligible | Reproduced | Reproduced per pair |
|---|---|---|---|
| G1 (γ 1.5) | ~1470 | ~950 | 950, 952, 948 |
| W (γ 2) | ~1580 | ~941 | 951, 926, 947 |
| G2 (γ 3) | ~1830 | ~957 | 968, 938, 965 |
| **H8 (W + downsampling 10)** | ~1695 | **~1004** | 1006, 995, 1012 |
| X1 (W + downsampling 5) | ~1830 | ~998 | 998 |

- **γ is on a plateau** from 1.5 to 3: reproduced units stay within W's own pair-to-pair spread, and
  γ 3's extra ~250 units carry about 0.06 bits/s each (5% of a typical unit). γ stays at 2.
- **`cluster_downsampling` 10 reproduces the most units.** Going to 5 adds ~135 eligible units and
  +6.8 bits/s (0.05 bits/s per unit, 4% of a typical unit) with no more reproduced units: fragments.
  10 is the operating point.

## What round 4 shows

- **Preprocessing is already right for this probe:** the high-pass at 150 Hz scores the same as
  300 Hz, 500 Hz loses 37 (it removes spike energy); whitening over 16 or 64 channels changes
  nothing that matters (−11, −6).
- **The universal templates aren't limiting:** more temporal shapes (+2) or wider footprints (−2)
  make no difference. Templates between the probe's columns (`dminx` 16) gain +13.4, just short of
  δ/2.
- **Landmark density in the first clustering pass matters:** with half the default downsampling,
  the clustering graph has twice as many landmark spikes per centre (up to the 25,000 cap), and it
  finds about 60 more units that reproduce across seeds. Halving again adds only fragments.

## Ablation of H8 (spec §9.3)

Each of H8's six changes reverted on its own; seed 1, plus seeds 2–3 where the cost was near δ/2.
Reverting `cluster_downsampling` gives W. Cost = H8's ΔS minus the reversion's, paired by seed.

| Reverted | To | ΔS | Cost | Eligible |
|---|---|---|---|---|
| `merge_bimodality_threshold` 0 (B1) | 0.6 | +68.0 | 323.7 | 875 |
| `merge_correlation_threshold_2` 0.95 (B3) | 0.5 | +263.4 | 128.3 | 1521 |
| `Th_learned` 7 (B5) | 8 | +354.3 | 37.3 | 1607 |
| `cluster_resolution` 2.0 (B2) | None | +368.6, +353.0, +365.3 | 23.0, 40.0, 28.8 (mean 30.6) | ~1340 |
| `Th_universal` 7 (B4) | 9 | +362.6 | 29.1 | 1624 |
| `cluster_downsampling` 10 (W) | 20 | +362.1, +384.4, +359.2 | 29.5, 8.5, 34.9 (mean 24.3) | ~1580 |

Every reversion passed every guard. None costs less than δ/2, so H8 is the simplified winner as it
stands, and its three replicates above are the winner's. With more landmark spikes, γ matters less
than it did on W (30.6 against 46.5), and so does `Th_universal` (29 against 46, one seed);
`Th_learned` matters slightly more (37 against 27, one seed).

## Budget

33 search runs in round 4 (30 screen, 3 extension) and 7 ablation runs; 12 ablation runs for round
3's leader. 176 sorts in all over four rounds (2 of them stopped), plus 54 re-scores of existing
sorts.
