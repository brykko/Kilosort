# KS2.5's interleaved template learning: how it works, how well, and why upstream left it

Lens of the interleaving audit (2026-11-04). The proposal: interleave extraction, clustering and template refinement over small batches in KS4, "somewhat like the template learning in KS2.5". Read-only: no sorts, no edits to kilosort or neuropixels.

**Labels.** *[code]* read in source. *[log]* measured from existing logs. *[paper]* authors' statement. *[inferred]* my reasoning. Quotes are under 15 words.

**Paths.**
- `LT` = `C:\code\github\neuropixels\ExternalToolboxes\KilosortToolbox\Versions\kilosort_v2.5\mainLoop\learnTemplates.m`; other `mainLoop\` and `CUDA\` files are in the same tree.
- `cfg` = `C:\code\github\neuropixels\+SpikeSort\+Kilosort\+Config\kilosort_config_v2_5.m`.
- `vX:path` = `git show vX:path` in `C:\code\github\Kilosort`.
- `SI` = KS4 Supplementary Information (Springer MOESM1, "Algorithms for Kilosort 2/2.5/3"), cited by its printed line numbers. `Paper` = Pachitariu et al., Nat Methods 21:914-921 (2024), from `C:\Users\richa\Downloads\s41592-024-02232-7.pdf`.

**Correction to the prior audit.** `modular_pipeline_audit.md` line 29 and §8 say stock KS2.5 is not on disk. It is: tags `v2.0`, `v2.5`, `v2.5.1`, `v2.5.2`, `v3.0` in this repo. Also, "KS2.5 sweeps batches in random order" is true of the lab copy only (§1.1).

---

## 0. Bottom line

1. **It isn't interleaved clustering in the sense the proposal pictures.** There is no clustering step: no features, no neighbours, no grouping of spikes. It is online hard-EM: match, average into the matched template, seed new templates from single residual snippets. The authors do call this clustering (SI 167-187), but it is k-means-like, with no way to split a template inside the loop. And KS2.5 is not interleaved end to end: a frozen whole-recording extraction and post-hoc merge/split/cut follow.
2. **For speed, the evidence cuts both ways.**
   - For: per-batch residual and match score reach about 98% of their final level within about 300 batches, under half a pass of a 20-min recording.
   - Against: as run, KS2.5 processes 4.4-6.2 pass-equivalents of a 20-min recording, about 1056 s against about 190-230 s for one KS4 cluster + extract round. Each batch costs a full matching pass (0.32 s against KS4's 0.24 s), so interleaving doesn't make a batch cheap.
3. **KS2.5's stopping point isn't a convergence measure.** Its stop at about 3000 batches is what its residual filter gives even if the residual is converged from the start (§2.3).
4. **Its yield is stable, but its identities are not.** Only batch order differs between seeds, and 69-75% of eligible units reproduce (KS4 default: 84-87%).
5. **Mixture templates come out of learning unsplit.** The first curation cycle makes 203-335 splits, 17-32% of the template count, and the merges before it remove another 154-225.
6. **Upstream never says online learning failed.** They say more powerful clustering can be applied to deconvolved features, and that the one-to-one burden on templates went away. Templates in KS4 only need coverage. An interleaver should copy KS2.5's update rule and residual-driven coverage, not its birth, gating or stopping.

---

## 1. The loop, exactly (Q1)

### 1.1 Three different versions

| | KS2.0 (`v2.0`) | Stock KS2.5 (`v2.5`) | Lab KS2.5, preset "standard" (what the harness runs) |
|---|---|---|---|
| Learning order | Half the recording from its middle out and back, in `rez.iorig` (depth-like) order (`v2.0:mainLoop/learnAndSolve8b.m:7,13`) | `rez.iorig`, which is `sort(mean(dshift,2))` (`v2.5:preProcess/datashift2.m:128`) unless `ops.midpoint` is set, then `randperm` (`v2.5:mainLoop/learnAndSolve8b.m:9-10,35`). Note: `v2.5:main_kilosort.m:41` says "RANDOM", and the datashift2 comment says "not really used any more", but it is used | Random: concatenated `randperm`s of valid batches (`learnAndSolve8b.m:7-19`) |
| Batches of learning | ≈ Nbatch | Nbatch, one pass (`v2.5:mainLoop/learnTemplates.m:51`) | Up to 16000 (`cfg:50`), auto-stopped (`LT:274-283`, `cfg:52`) |
| Momentum m (spikes) | linspace 20→400 by batch | linspace 20→400 by batch (`v2.5 LT:62`) | Per template: m = clip(0.1 × nspTot, 5, 400) (`LT:150-162`, `cfg:31,48`) |
| pm = exp(-1/m) | 0.9512→0.9975 | same | 0.8187→0.9975; saturates at nspTot = 4000 |
| nsp filter p1 | 0.95 (τ≈20 batches) | 0.95 (`v2.5 LT:84`) | 0.99 (τ≈100 batches) (`cfg:22`) |
| minFR | 1/50 Hz | 1/50 Hz | 0.05 Hz (`cfg:32`) + `templateMinSpikes` 50 at the end (`cfg:25`) |
| Similarity drop | corr > 0.9 | corr > 0.9 (`v2.5:mainLoop/triageTemplates2.m:32`) | corr > 0.975 (`cfg:28`) |
| Th | [10 4] | [10 4] | [10 6]; standard_low [9 4] (`cfg:49,57`) |
| Matching iterations | 60 | 60 (`v2.5 LT:57`) | up to 500, breaks when no new spikes (`cfg:51`, `mexMPnu8_rg.cu:966,1063`) |
| Final pass | Templates keep updating (`v2.0:mainLoop/trackAndSort.m:164-166`) | Frozen (`v2.5:mainLoop/trackAndSort.m:163` only accumulates) | Frozen (`trackAndSort.m:117,182`) |

The lab's changes all lengthen or stabilise learning. `cfg:94-95`: "original_hq", increased optimisation time, "likely to work better" for short recordings. Commit `e4d5cc5` (2025-07-06) added the convergence stop. So the lab copy of KS2.5 is not upstream's learning schedule: upstream's was one pass.

### 1.2 One batch of learning (lab copy, `LT:120-404`)

1. Load one batch: 65,600 samples, 2.19 s (`LT:123`).
2. Set per-template pm from nspTot (`LT:146-165`).
3. Re-sort every template-indexed array by peak channel, for GPU memory locality (`LT:169-183`).
4. **Refresh the factorisation, every batch, for every template.** `mexSVDsmall2` takes dWU, blanks it to the 32 channels around the peak, power-iterates a rank-3 temporal basis from a warm start (100 iterations, `mexSVDsmall2.cu`, `getW`), gets U, re-normalises to W, U, mu, and aligns the trough (`LT:188`, `mexSVDsmall2.cu:341-353`).
5. Boolean overlap mask UtU, every batch (`LT:195`, `getMeUtU.m:15`). This is channel sharing, not correlation. No template cross-correlation table is built during learning: `getMeWtW` runs once at the end (`LT:418`), and the in-loop `getMeWtW2` (zero lag only) runs only in triage (`triageTemplates2.m:32`). The matcher instead re-filters raw data around each subtracted spike (`spaceFilterUpdate`/`timeFilterUpdate`, `mexMPnu8_rg.cu`, loop at 966).
6. Matching pursuit at Th(1) = 10 on the whole batch against the whole dictionary (`LT:215-217`). Observed 49-80 iterations per batch on kobe r1 [log]. Score: Cf = b²/(1+λ) − λμ², b = max(0,proj) + λμ, λ = 10 (`mexMPnu8_rg.cu:310-312`). A spike is taken if Cf > Th² (`:409`). Spikes below Th(1) stay in the residual.
7. Batch means, deconvolved. For each spike above Th(1), sum residual + its own template (`mexMPnu8_rg.cu:691,703`). So overlaps are already subtracted, and templates are means over only those spikes above Th(1) (a truncated sample, so low-amplitude units are biased upward [inferred]).
8. **Template update, count-based** (`updateTemplates.m:5-8`): fexp = pm^n; dWU ← fexp·dWU + (1−fexp)·(batch mean). Authors: about 400 spikes for a template to forget its value (SI 68-70).
9. Spike-count filters: nsp EMA p1 (triage), nsp2 EMA p2 (`LT:266-267`), and a running total (`LT:269`).
10. Auto-stop check (§2.3).
11. **Triage every 5 batches** (`LT:290-294`, `triageTemplates2.m`).
    - Drop if nsp < m0 (rate gate). m0 = 0.05 Hz × 2.19 s = 0.109 spikes/batch (`LT:114`).
    - Drop the lower-count one of pairs with zero-lag corr > 0.975 and |μ difference| < 12.65 (`triageTemplates2.m:43-54`). This is a drop, not a merge: nothing is averaged (also SI 236-237).
12. **Birth, every batch** (`LT:308-347`). `mexGetSpikes2` runs on the residual:
    - Six prototype waveforms over up to 7 channels; score Cf²/(1+j) > Th(1)² (`mexGetSpikes2.cu:22-55,155`).
    - Non-maximum suppression within ±4 channels over the *whole batch* (`mexGetSpikes2.cu:204`, SI 219-221). So at most one per 5-channel neighbourhood per batch: a cap of about 77. Batch 1 gave 61-77 [log].
    - Each birth is one raw residual snippet, projected on 3 PCs (`LT:319`), appended with nsp = m0, nspTot = 1, mu = 10, pm for m = 5 (`LT:325-330`). Clipped at Nfilt.

**Final steps.** Triage with `nspTot ≥ 50` (`LT:414-415`), `getMeWtW`, memorise W/U/mu. Then `runTemplates`→`trackAndSort` over 1:Nbatch at Th(2), frozen, with features (`runTemplates.m:32-34`, `trackAndSort.m:70-81`). Then `autocurate` (merge/split/cut twice, then merge). No split exists in `LT` or in `v2.0` or `v2.5` learnTemplates (SI 156-157 says splits occur in the KS2 learning phase; I found none there; the code has merges and splits only post hoc).

**Costs.** Every batch is a full matching against every template: 0.32-0.61 s/batch in learning at about 1000 templates, against 0.37-0.87 s/batch in final extraction [log]. KS4's learned-template extraction is about 0.24 s/batch (143.5 s over 600 batches of 60,000 samples, `C:\data\ks4_tuning¶48_kobe_2020-08-21uns\hybrid_default_seed1.log`). KS2.5 keeps per-batch overhead tolerable by never building a cross-correlation table. KS4 builds `ctc` once and depends on it being static (`kilosort/template_matching.py:68,161-184`). Changing templates in KS4 re-opens that cost [inferred].

---

## 2. How it behaves (Q2)

**Method.** Parsed every `ks25_sort.log` under `C:\data\ks4_tuning` (18 distinct sorts; `strict_cleaning` and exact duplicates excluded). The log gives one row per 100 batches. `mdres` and `mvexp` are EMAs with p2 = 0.998 (τ = 500 batches, `LT:94,256-258`), so the printed curves lag badly. I deconvolved them: `(EMA_i − 0.998^100·EMA_{i−100}) / (1 − 0.998^100)` gives approximately the mean over the previous 100 batches. `avgmdres` is a median over channels of per-channel EMAs, so this is approximate. No W snapshots exist, so per-template change cannot be measured from these data (§2.5).

### 2.1 Passes and stopping [log]

| Data | Sorts | Stop batch (min / median / max) | Passes of the recording |
|---|---|---|---|
| 20-min excerpts, 549 batches | 14 | 2431 / 3050 / 3419 | 4.43-6.23 |
| 2-h segments, 3293 batches | 4 | 2509 / 3018 / 3648 | 0.76-1.11 |

On kobe "standard" r1: 3251 batches matched about 29 M spike instances at Th 10 (about 9000/batch), against 8.06 M extracted spikes at Th 6 (`stage_breakdown.log`). Learning 1056 s, extraction 205 s, an order-of-magnitude comparison on a shared machine.

### 2.2 Aggregate fit plateaus fast [log]

kobe standard r1 (MEC and the 2-h kobe segment are the same shape):

| Batch | Units (post-triage) | Sum nsp | Median mu | mdres EMA | mdres, last 100 (deconvolved) | mvexp EMA | mvexp, last 100 | Births/batch |
|---|---|---|---|---|---|---|---|---|
| 101 | 708 | 4582 | 15.66 | 0.9890 | 0.9412 | 19.01 | 14.77 | 38.7 |
| 201 | 791 | 7000 | 15.23 | 0.9796 | 0.9372 | 18.08 | 13.88 | 37.7 |
| 301 | 834 | 8063 | 14.90 | 0.9717 | 0.9361 | 17.32 | 13.89 | 37.2 |
| 501 | 883 | 8596 | 14.55 | 0.9599 | 0.9351 | 16.16 | 13.77 | 37.2 |
| 1001 | 937 | 9085 | 14.26 | 0.9447 | 0.9348 | 14.65 | 13.79 | 36.4 |
| 2001 | 1002 | 9110 | 13.99 | 0.9367 | 0.9344 | 13.88 | 13.74 | 35.8 |
| 3001 | 1047 | 9076 | 13.86 | 0.9347 | 0.9333 | 13.79 | 13.79 | 34.9 |

- First-batch residual 0.9996, plateau 0.9349 (last-ten SD 0.0012). So the mean over batches 2-101 is already 90% of the way down, batch 201 is 96.5%, batch 301 is 98%.
- Per-batch match score is flat by batch 200.
- **So per-batch fit is at about 98% within about 300 batches: 11 min of data, 0.55 pass of a 20-min recording.** The same on the 2-h segment (0.09 pass), which says it scales with batches seen, not with recording fraction. MEC: within 0.005 of plateau by 201, slower tail to about 0.943.
- This is an aggregate of big, frequent units. Whitened residual falls only 6.5% in total; it is mostly noise. It would barely notice a mixture template or a missing rare unit [inferred].

### 2.3 The auto-stop batch measures filter lag, not convergence [log + simulation]

Rule: stop when EMA(i) ≥ EMA(i−500) (`LT:274-277`, `cfg:42`). The EMA starts at 0.9996 (`LT:251-253`) and takes `ln(gap/noise)` time constants of 500 batches to fall into its own noise, whatever the data does.

Simulation (scratch, not kept): residual steps instantly to plateau plus white noise, σ = 0.011 (the per-batch noise implied by the last-ten SD), same EMA and rule, 60 runs. Stop at median 2923 (range 2377-3935). Adding τ = 60 batches of real decay gives 2978. Observed: median about 3020, range 2431-3648, in both 20-min and 2-h data.

So the constant stop batch across a 6× change in recording length is not evidence that about 3000 batches are needed. Don't read it as a convergence time, and don't copy the rule.

### 2.4 Template population converges slowly [log]

- **Count.** Over the 18 sorts, at the post-triage printed rows: 50% of the final count by about batch 100, 75% by 200-400, 90% by 700-1600, 95% by 1300-2400. Still rising 2-9 units per 100 batches at stop. **It never plateaus.**
- **Age.** At stop, half of the pool was born by batch 91 (kobe r1), 232 (MEC r1), 82 (2-h kobe); a quarter by batch 13 on kobe r1 (`lt` percentiles, `LT:364`). Lineage only: contents are replaced by the running average.
- **Momentum.** Median template reaches 4000 spikes (m = 400) by batch about 900 on kobe r1; lower quartile about 1800 (`nstot`, `LT:365`). A unit's learning time is m/(rate × 2.19 s) batches: a 0.1 Hz unit (spikes above Th 10) needs about 18,000 batches to see 4000 spikes, or about 180 to see 40 spikes. On a 20-min recording that same spike is seen again each pass. No batch schedule changes this: rare units are limited by spike count [inferred].
- **Churn does not converge.** Births stay at 35-39 per batch from first to last batch, near-constant, close to half the cap. Summed over a run: 86,680-142,573 births for about 1000 templates kept, about 1% survival. Between triages, up to about 175 probationary templates sit in the dictionary (five batches of births). The stop row of kobe r1 shows it: 1188 in the pool against 1054 after the triage printed at batch 3201. Final triage then removes 64-219 templates (986 kept of 1188 on kobe r1), which suggests a sizeable share of the pool is transient [inferred; the log doesn't split that count by reason].
- **Similarity drop is almost idle.** `ndrop(2)` prints 0.016, about 0.02 per triage once its EMA is corrected, at corr > 0.975 [log]. In the lab preset online merging essentially doesn't happen.

### 2.5 Late learning still adds score: a natural experiment [log, post hoc]

Stop time is mostly noise (§2.3), so across seeds it nearly randomises learning length. Within preset × excerpt, across 12 sorts (three per cell):

| Slope per 100 extra batches | Estimate | p (7 df) |
|---|---|---|
| S | +3.2 ± 1.5 | 0.08 |
| Eligible units | +4.9 ± 1.3 | 0.008 |
| Templates | +4.7 ± 1.8 | 0.03 |

**Bias.** I chose this after seeing the logs, n = 12, and S is a noisy outcome. Read it as a hint. It agrees with the growth in §2.4 and argues that residual plateau at batch 300 does not mean the dictionary is finished: 700 extra batches ≈ +22 S, about three quarters of δ (28.6).

**Cheap direct test of convergence.** `ks25_sort.m`'s `testOps` can set `initNBatch` to 300, 549, 1000, 2000 with no code edit. Score S per truncation. This is the measurement this audit can't make. Snapshots of W every 100 batches would answer "how much do templates still change", but need a code hook.

### 2.6 Reproducibility [log]

The seed changes only batch order: `learnAndSolve8b.m:7`; `rng(LTseed)` is fixed at `LT:13-14`, and `trackAndSort.m:31` is fixed. A rerun with the same seed reproduces every spike (`ks25_comparison.md:366-368`). So cross-seed differences are **pure order dependence**, through the whole of learning and curation.

| Sort | Units matched across seeds (agreement ≥ 0.5, `reproducibility.py`) |
|---|---|
| KS2.5 standard, kobe | 0.69-0.72 |
| KS2.5 standard, MEC | 0.73-0.75 |
| KS2.5 standard_low, kobe / MEC | 0.65-0.70 / 0.71-0.75 |
| KS4 default, kobe (RNG in clustering) | 0.84-0.87 |
| H8, kobe | about 0.59 of a much larger unit set (`ks25_comparison.md:86-94`) |

S itself is steady: 982.6-994.2 on kobe (SD 5.9), 2032.8-2066.9 on MEC (`ks25_comparison.md:57-60,110-114`). So order changes *which* units, not *how many*. Stop batch varies with seed too (kobe 2492-3251).

---

## 3. Failure modes (Q3)

- **Order dependence.** §2.6. Authors: for learning, batches go in pseudorandom order, "For learning to be effective" (SI 70-72). KS2 gave that up for tracking. Random order makes it a stochastic optimiser, with its variance.
- **Drift tracking and why datashift + learnAndSolve exist.** KS2 fixed the batch order so the running mean would follow drift (SI 73-78). KS2.5 made that unnecessary: registration "corrects the raw data directly" (`v2.5:README.md`), then `datashift2` ends "we can just run a normal spike sorter" (`datashift2.m`, final comment). The Neuropixels 2.0 methods say the drift tracking step wasn't needed, and templates came from pseudo-random batches, then were "frozen to their final values" (via WebFetch summariser on the bioRxiv text; that sentence was returned verbatim, the rest paraphrased; I could not read it directly). So learnAndSolve survives in 2.5 because it was the template learner, not because forgetting is needed [inferred]. Fixed forgetting (m = 400) on registered data throws away spikes: the dictionary ends as an EWMA of about the last 400 spikes of a template, in random batches. Authors also say tracking fails with fast drift because a slow neuron gives "very few spike samples" (SI 84-87), and that it needs a continuous drift range, not step drift (`Paper` p.918; SI 16-19).
- **Birth/death instability.** §2.4: churn never falls. Templates are killed on an EMA of rate and reborn from single spikes. The Kilosort wiki ("How it all works", via summariser) describes this turnover as candidates being tried on.
- **Spike-count gating of sparse units.** A newborn has nsp = m0 and, with no spike, falls below it at once. It needs a spike above Th(1) before its first triage, 1-5 batches later (nsp0 ≥ m0 means ≥ 1 spike; `LT:266,114`). P(spike in 5 batches) is 66% at 0.1 Hz, 42% at 0.05 Hz, 20% at 0.02 Hz; if born one batch before triage, 20%, 10%, 4%. Afterwards it needs about 1 spike per 9 batches, and it must reach 50 total. All counted at Th(1), not Th(2). A unit whose spikes rarely exceed 10 fails however high its true rate is [inferred from code].
  - Empirically, after curation: < 1.2% of KS2.5 units fire < 0.1 Hz, 5th percentile about 320 spikes in 20 min on all four sorts checked. Of H8's below-median eligible units KS2.5 finds 0-0.3% (`iteration\ks25_found_rate.txt`), though Th, λ and the cut also contribute: that isn't attributed to gating alone.
  - The wiki says sparsely firing neurons are hard to track.
- **Dependence on Th(1), Th(2), λ.** Th(1) sets matching, averaging and birth at once (Params 3 and 16, `LT:66-67`). With λ = 10 the threshold is amplitude-relative: for μ = 14 proj must exceed about 10.5; μ = 30, 16.4; μ = 50, 25.4 [computed from `:310-312`]. The score is thus partly membership. standard_low moves S by +168 on MEC and about +85 on kobe (`ks25_comparison.md`), so learning-stage thresholds are strong levers. The score is also what `set_cutoff` later thresholds (prior audit §4).
- **Splitting afterwards; convergence to mixtures.** Online updates never split. A mixture has a stable mean; both units match it; residual stays below birth threshold [inferred]. Authors: "Splits however are much more difficult" (SI 242-243), merges easy. Observed first-cycle splits: 260-266 on kobe, 203-206 on MEC, 335 on kobe low, from 970-1182 templates; second cycle 37-79. Merges 154-225 then 49-89. Splitting moves plain misassignment 23.8% → 20.2% (kobe), 34.3% → 31.7% (MEC) (`stage_breakdown.log`). Over-split also occurs: 178-225 merges in cycle 1. Authors designed global moves "to escape local minima" (SI 171-173).
- **Length effect.** The 2-h kobe sort: +9.5% S but +54% misassigned (prior audit §5), untraced. Note that batches of learning stay near 3000 while spikes rise 6×, and final templates only 967→1147. One candidate; untested [inferred].

---

## 4. Why upstream moved (Q4)

**Stated.**
- KS3 README (`v3.0:README.md`): the main change from 2.5 is "a completely new and much more sophisticated clustering algorithm".
- `Paper`, Results: KS4 templates are "completely discarded after spike extraction", "because more powerful clustering algorithms can be applied" to the features.
- `Paper`, Methods, "Template learning": in versions 1, 2, 2.5, "special care was taken" that templates match neurons one-to-one, "necessary because relatively few additional merges and splits were performed". In 3 and 4, templates need not match single neurons, but "every spike in the raw data has some template to match to".
- `Paper`: KS4 "consistently performed better due to its improved clustering algorithm" (p.918). KS2.5, 3, 4 have "the same template deconvolution strategy but differ in the clustering".
- `Paper` ablation: reclustering after deconvolution had "a smaller but consistent effect". Some steps "redundantly fix problems left over by the other steps".
- SI opening: these algorithms are "no longer used in Kilosort4". KS3 grew from bimodal pursuit working unexpectedly well on mixtures (SI 341-347).
- `Paper` intro: Kilosort1 "combined two steps" (detection + clustering = template learning). Descriptive, not a criticism.

**Not stated anywhere I could find.** That online learning was at fault. In the KS4 main text the word "online" appears only in publisher boilerplate (checked by grep over the full PDF text). No statement about convergence speed, order dependence, or reproducibility of online learning. `v2.5..v3.0` git messages give no reason (`b1cca4d`: "this is Kilosort3!"). Issue searches found no maintainer discussion of this; the search tool isn't exhaustive.

**My inference, not theirs.** The driver was the wish for better clustering, not dissatisfaction with online learning as such. But one-to-one templates made splits the bottleneck, and the online loop can't split; moving clustering out of the template stage removed that burden. Two-stage has existed since the first KS3 commit (`v3.0:main_kilosort3.m:40-46`, Jan 2021). Upstream chose whole-recording clustering and has never returned to interleaving, but no paper shows the interleaved route fails. Absence of an authors' verdict is not evidence for the proposal either: they never compared the two designs with the same clusterer.

KS4's attribution is itself under-identified (prior audit §5).

---

## 5. What a KS4 interleaver should take (Q5)

**How KS2.5 differs from the picture.**

| Pictured | Actually |
|---|---|
| Extraction interleaved | Matching at Th(1) = 10 only, no features; the real extraction is a separate frozen pass at Th(2) |
| Clustering interleaved | None. Hard assignment by best match; spikes never reassigned, except by re-visiting a batch |
| Template refinement interleaved | Yes: count-based running mean, every batch |
| New units from clusters | One residual snippet each, about 35 per batch, about 1% survive |
| Fast | About 300 batches for aggregate fit; 4.4-6.2 passes as run on 20 min |
| Self-contained | Needs post-hoc merges, splits on the scale of 17-32% of templates, `set_cutoff` |

| Mechanism | Take? | Why |
|---|---|---|
| Count-based update per template (`updateTemplates.m`) | **Yes**, as a running mean | Exact mean on registered data; the lab's move from batch to spike annealing (`LT:150-162`) was the right change. Forgetting isn't needed after registration; a soft start for early mixtures may be |
| Deconvolved averaging: residual + own template (`mexMPnu8_rg.cu:703`) | Already in KS4 | Keep it for any online update |
| Residual as a source of new coverage | **Yes, in idea** | Matches KS4's stated need (every spike has a template). Coverage is what KS4 templates are for |
| Single-snippet birth at per-batch cap | **No** | 99% wasted; adds Nfilt load; `ctc` cost in KS4 would suffer. Buffer residual spikes and cluster them with KS4's own clusterer [inferred] |
| Random batch order on registered data | **Yes** | Authors' condition for online learning (SI 70-72); drift-sorted order was for tracking |
| Whole-pass guarantee | **Yes, keep** | Rare units are spike-limited (§2.4), and three of the four 2-h runs stopped before finishing a pass (0.76-0.92) |
| Usage-based pruning | Maybe | On cumulative counts over a whole pass, not an EMA of rate; the EMA gate is what hurts sparse units |
| Per-batch SVD of all templates | Probably not | KS4 templates are in a PC basis; a running mean in that basis needs no SVD, though low-rank U does |
| Local re-filtering instead of a `ctc` table | Unclear | KS2.5 avoids `ctc` but pays per subtraction; KS4's speed relies on static `ctc`. Rebuild `ctc` per update round, not per batch [inferred] |
| Zero-lag similarity drop | **No** | Drops, doesn't merge; almost idle at 0.975; KS4 has lag-aware template merging |
| EMA residual auto-stop | **No** | Measures its own lag (§2.3). Use a held-out or unsmoothed measure |
| Forgetting at 400 spikes, Th(1) gating of averages, template = unit | **No** | Tracking relics or the one-to-one burden KS4 dropped |

**Bearing on the hypothesis.**
- **Supports:** aggregate fit arrives in under half a pass; dictionaries grow most in the first 100-400 batches.
- **Cuts against:** per-batch cost is about a matching pass, so saving must come from visiting fewer batches, but rare units need spikes, and KS2.5 keeps gaining units through 3000 batches. Order dependence is high (§2.6). Mixtures survive the loop.
- **Key gap:** no one has measured dictionary quality against batches. The truncation test (§2.5) is cheap and should come before building anything. A KS4 round is about 190-230 s, roughly 600-700 KS2.5 batches, so an interleaver would need to reach a round's benefit in well under a pass to be faster.

## 6. Unverified or limited

- No W snapshots: "how much templates change late" is unmeasured. All curves are aggregates.
- Deconvolution assumes per-100-batch sampling and an EMA of medians; approximate.
- `Steinmetz 2021`: bioRxiv full text via WebFetch summariser; accepted PDF returned 403. Only one quoted sentence was verbatim.
- Wiki statements via summariser, not read directly.
- PMC page was blocked; the journal PDF and MOESM1 were read locally.
- GitHub issues were searched, not read exhaustively.
- Timing compares runs under different loads.
- Spike-gating probabilities assume Poisson firing with all spikes above Th(1).
- Reproducibility includes curation, not learning alone.
