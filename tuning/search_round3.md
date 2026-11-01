# Search round 3: detection and features

Plan for round 3, frozen before its first run. Builds on [evaluation_spec.md](evaluation_spec.md) (v0.13), whose score,
guards, information barrier and selection rules apply unchanged, and on
[round2_results.md](round2_results.md).

## 1. Aim

Rounds 1 and 2 tuned how spikes are clustered and merged; nothing upstream has been touched. Round 3
asks, on top of the round 2 leader, whether KS4 finds and represents spikes as well as it could:

- **Detection:** which spikes are found at all. Lower thresholds find more spikes from small and
  distant cells, which S can reward (more informative spikes) or which can add noise and duplicates
  (the guards' job).
- **Features:** what the clustering sees for each spike: how many channels (`nearest_chans`) and how
  many temporal components per channel (`n_pcs`). Richer features could separate cells that the
  current 10 × 6 features can't.
- **Drift:** non-rigid drift correction, in case residual drift stretches units across depth.

Base: **R4** (bimodality 0, γ 2.0, first merge 0.975, final merge 0.95), mean ΔS +299.5 over three
seeds; ΔS below is against default, as before.

## 2. Preparation (not counted)

- **Does `max_peels` bind?** One R4 sort, seed 1, with a hook counting, per batch, the peels template
  matching runs before it finds no more spikes. If no batch reaches the limit of 100, D10 is
  dropped. Behaviour-blind.
  **Result** (`tuning/peel_diagnostic.py`, `runs/round3_peel_diagnostic.json`): 533 of 600 batches
  (89%) stop at the limit of 100 peels, so matching leaves spikes above threshold undetected in most
  of the recording. D10 stays.
- **Hybrid units and drift:** inserted units follow the reference sort's rigid drift estimate
  (spec §7.3). With non-rigid correction (D9) their drift no longer matches what KS4 corrects, so
  hybrid recovery may drop for reasons unrelated to sorting quality. D9's hybrid result is reported
  with that caveat.

## 3. Screen (R4 + one change, seeds 1–3)

| Run | Change from R4 | Default | Question |
|---|---|---|---|
| D1 | `Th_learned` 7 | 8 | Does matching more small spikes add information? |
| D2 | `Th_learned` 6 | 8 | Further |
| D3 | `Th_learned` 9 | 8 | The other direction: fewer, cleaner spikes |
| D4 | `Th_universal` 8 | 9 | Do more first-pass detections give templates for more cells? |
| D5 | `Th_universal` 7 | 9 | Further |
| D6 | `nearest_chans` 14 | 10 | Wider features for clustering |
| D7 | `nearest_chans` 20 | 10 | Further |
| D8 | `n_pcs` 9 | 6 | More temporal components per channel |
| D9 | `nblocks` 5 | 1 | Non-rigid drift correction |
| D10 | `max_peels` 200 | 100 | The limit binds in 89% of batches (§2): are spikes being missed? |
| | **Total** | | **10 configurations × 3 seeds** |

These change stages the stage cache can't reuse (detection and drift), so each run is a full sort,
about 5 minutes: roughly 2.5 hours for the screen.

## 4. Leader and combination

As in round 2:

- **Leader:** starts as R4. A configuration becomes leader if its 3-seed mean ΔS exceeds the
  leader's by ≥ δ/2 (14.3), it passes every guard on every seed, its mean ΔS is ≥ δ and its mean
  ΔS_track is ≥ δ_track/2.
- **Combination:** settings that improve on R4 by ≥ δ/2 with all guards passing are added to the
  leader one at a time, best first, each on three seeds; an addition is kept if it becomes leader.
- **Resolution:** detection and feature changes can also change how many units come out. For any
  candidate leader, report the units reproduced across seeds (`reproducibility.py`); if a change
  works mainly by adding units, the spec's resolution stopping rule (§9.2) applies.

## 5. End of round 3

Report the leader with its replicates, a table of every configuration (ΔS, ΔS_track, eligible
units, hybrid recovery, cleaning quantities, guard failures, reproduced units for candidates), and
proposals. Then stop for discussion.

Left for later: whitening range, high-pass cutoff, template-size settings for universal templates,
and explicit `x_centers`/`dmin` (clustering group geometry).
