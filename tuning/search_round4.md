# Search round 4: preprocessing, universal templates, resolution check

Plan for round 4, frozen before its first run. Builds on [evaluation_spec.md](evaluation_spec.md)
(v0.14), whose score, guards, information barrier and selection rules apply unchanged, and on
[round3_results.md](round3_results.md) and the ablation of its leader.

## 1. Aim

Round 3 found that detection was the biggest lever left: lowering `Th_universal` to 7 gave templates
to more cells (+44 over R4), and lowering `Th_learned` to 7 added spikes to existing ones (+27 more).
Both saturate at 7. Round 4 asks what else limits detection, and rechecks the resolution choice on
the new base:

- **Preprocessing:** what the detector sees. The high-pass cutoff sets how much of each spike's
  energy survives filtering; the whitening range sets how much of the shared noise between channels
  is removed.
- **Universal templates:** the shapes, sizes and positions the first detection pass tries. A spike
  that matches none of them well is detected late or not at all, and a cell without first-pass spikes
  gets no template.
- **First clustering pass:** `cluster_downsampling` 10 (more landmark spikes per clustering centre)
  added +35 on `standard` in round 2 and was never tried on R4.
- **Resolution recheck:** γ 2 was chosen on R4 by the stopping rule (spec §9.2). The new base
  detects more spikes and more cells, so the γ with the most reproduced units may have moved.

Base: **W**, round 3's leader C1 simplified by ablation (spec §9.3; [round3_results.md](round3_results.md), Ablation):
bimodality 0, γ 2.0, final merge 0.95, `Th_universal` 7, `Th_learned` 7, with the first merge back
at its default (0.9). Mean ΔS +368.6 over three seeds; ΔS below is against default, as before.

## 2. Screen (W + one change, seeds 1–3)

| Run | Change from W | Default | Question |
|---|---|---|---|
| H1 | `highpass_cutoff` 150 | 300 | Does keeping more low-frequency spike energy help detection? |
| H2 | `highpass_cutoff` 500 | 300 | The other direction: less LFP and slow noise |
| H3 | `whitening_range` 16 | 32 | More local whitening |
| H4 | `whitening_range` 64 | 32 | Broader whitening: more shared noise removed |
| H5 | `dminx` 16 | 32 | Universal templates halfway between the probe's columns as well as on them (the grid is already half-pitch along the probe) |
| H6 | `template_sizes` 8 | 5 | Wider spatial footprints (up to 80 µm) for large cells |
| H7 | `n_templates` 9 | 6 | More temporal shapes for the first detection pass |
| H8 | `cluster_downsampling` 10 | 20 | More landmark spikes in the first clustering pass |
| G1 | `cluster_resolution` 1.5 | (2.0 in W) | Resolution recheck, below |
| G2 | `cluster_resolution` 3.0 | (2.0 in W) | Resolution recheck, above |
| | **Total** | | **10 configurations × 3 seeds** |

H1–H7 change preprocessing or detection, so each is a full sort (about 6 minutes); H8, G1 and G2
reuse W's cached drift correction and first detection pass. About 3 hours for the screen.

H5 roughly doubles the number of universal templates (about 2700 instead of 1540); its detection
memory (about 2.7 GB per batch) fits on the GPU.

## 3. Leader, combination, resolution

As in round 3:

- **Leader:** starts as W. A configuration becomes leader if its 3-seed mean ΔS exceeds the leader's
  by ≥ δ/2 (14.3), it passes every guard on every seed, its mean ΔS is ≥ δ and its mean ΔS_track is
  ≥ δ_track/2.
- **Combination:** settings that improve on W by ≥ δ/2 with all guards passing are added to the
  leader one at a time, best first, each on three seeds; an addition is kept if it becomes leader.
- **Range extension:** if a setting's best level is the end of its range and it improves on W by
  ≥ δ/2, one further step in the same direction is run on three seeds.
- **Resolution:** G1 and G2 are judged by the stopping rule (spec §9.2), not by S: units reproduced
  across seeds (`reproducibility.py`, seeds 1–2) for G1, W and G2. γ moves only if G1 or G2
  reproduces clearly more units than W (more than the seed-pair spread, about 10). For any other
  candidate leader, reproduced units are reported, and a change that works mainly by adding units
  falls under the same rule.

## 4. End of round 4

Report the leader with its replicates, a table of every configuration (ΔS, ΔS_track, eligible
units, hybrid recovery, guard failures, reproduced units for candidates), and proposals. If the
leader changes, it goes through ablation (spec §9.3) and the stricter-cleaning check (§9.2) before
the final report. Then stop for discussion.
