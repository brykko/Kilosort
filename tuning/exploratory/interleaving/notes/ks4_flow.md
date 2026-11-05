# KS4 flow and where interleaving could go in

Lens: how KS4 and its multi-round extension work, and where finer interleaving of extraction, clustering
and template refinement could or could not be inserted. Builds on `..\..\modularity\modular_pipeline_audit.md`.

Code: worktree `.claude\worktrees\ks4-round3`, tip a74cbf9. Citations are `file:line` from there
(`rk` = `kilosort/run_kilosort.py`, `tm` = `template_matching.py`, `sd` = `spikedetect.py`,
`cq` = `clustering_qr.py`, `ss` = `swarmsplitter.py`).

**[V]** = read in the code, or read directly from saved logs and files. **[M]** = measured here (§7). **[I]** = inferred.

## 0. Findings that matter most

1. **The speed-up has a hard cap of about 3×.** [V from logs] Drift, the final extraction, the final clustering, merge
   and save cost 265 of 826 s (20 min, H8, four rounds) and about 1975 of 6005 s (2 h). Learning, which
   interleaving could replace, is 67-68%. Going below the cap would mean changing the final passes.
2. **Features carry the spike's own waveform, not the template.** [V] `tm:90-91`. Round 1 features don't use
   the universal template at all (`sd:285-287`). A poorly fitted spike that crosses a threshold reaches
   clustering intact.
3. **The user's premise holds only at one gate, and that gate is mostly not about waveform fit.** [M, estimate] The
   universal pass finds 45% of what round 2 matches. A local-maximum rule alone would hide about 44% of matched
   spikes (`sd:176, 181-182`), which accounts for 71-81% of the shortfall. At most about 11% of matched spikes
   (kobe H8; 16% at default) are left to blame on poor fit, and that residual also includes false positives and echoes.
4. **Extra rounds find few new spikes.** [V, `round3_results.md`] +1.5% then +0.7%. The gain is thin, wide
   cleaning of existing units. So a design justified by discovering missed waveforms aims at the wrong effect.
5. **`prepare_matching` is not the obstacle.** [M] A full `ctc` rebuild takes 0.024 s on 1761 real templates
   against about 0.17 s per batch for matching. The expensive and data-hungry part is clustering: about 0.09 s
   per centre even on tiny buffers, so 8-10 s per sweep of 96 centres, 50× a batch of matching.
6. **The existing clustering cannot run on a batch.** [V + arithmetic] A 5 Hz unit needs 11-22 batches at the very
   least to reach the graph floor (25-50 in round 1, where only about 45% of spikes are seen). Pairs at 1 Hz × 1 Hz
   need 125 batches before the tree's refractory test is more than a shortcut. Interleaving with this
   clustering therefore means rounds on tens to hundreds of batches, which is D3, not KS2.5.
7. **D3's seam already exists, with a trap.** [V] `batch_downsampling` strides batches, but `dshift` is indexed by the
   unstrided counter (`io.py:893, 1137, 1159-1160`). `bfile.set_downsampling(k)` on the corrected file silently
   applies the wrong drift. Building a second file with `dshift=ops['dshift'][::k]` avoids that.
8. **Clustering restarts from scratch every round.** [V] `cq:531` never passes the `iclust` that `cluster()` accepts
   (`cq:125, 167-172`). Round 3 replaces about as many units as a seed change does. [I] Some of what looks like slow
   convergence may be restart noise. Warm starting (D5) is a cheap untested alternative.

## 1. State through the stages

| Stage | Call | Per batch | Needs whole recording | Output, and what is held |
|---|---|---|---|---|
| Preprocess | `rk:660`, `compute_preprocessing` | | whitening over strided batches | `whiten_mat`, `hp_filter`, `ops['Nbatches']` |
| Seed | `rk:285-288` | | | numpy, torch, cuda RNG |
| Drift | `datashift.run` → `sd.run` (`datashift.py:198`) | universal detection on uncorrected data; `tF` discarded | `bin_spikes`, `align_block2` (`datashift.py:204-207`) | `dshift` (Nbatches × blocks), `yblk`, `iKxx`; side effects `wPCA`, `wTEMP`, `ycup`, `xcup`, `dmin`, `dminx`, `iC`, `iC2`, `weigh` (`sd:119-120, 223, 250, 319-321`) |
| Round 1 detect | `rk:873` → `sd.run` | same, on corrected data; `wPCA`/`wTEMP` recomputed (`sd:220-227`) | | `st0` 6 columns (time s, y, amp, imax, batch, position), `tF` (n, 10, 6) raw |
| Round 1 cluster | `rk:899`, `cq.run(mode='spikes')` | | per-centre, all spikes | `clu`, `Wall` (n_clu, Nchan, 6), CPU |
| Dictionary | `rk:905`, `postprocess_templates` | | `Wall`, cluster sizes only (`tm:167-171`, mode 'mu') | `Wall3` = U (n, 6, Nchan). `st0`, `tF` freed (`rk:925`) |
| Round 2 extract | `rk:926` → `tm.extract` | `run_matching`, features (`tm:88-113`) | `ctc` (`tm:68`) | `st` 3 columns (sample, template id, score), `tF`; sets `iCC`, `iCC_mask`, `iU` (`tm:62-64`) |
| Round 2 cluster | `rk:349`, `cq.run(mode='template')` | | per-centre plus correlograms | `clu`, `Wall` |
| Extra rounds | `rk:354-392` | | `_next_templates` then extract and cluster; previous `st`, `tF` freed (`rk:380`) | U replaced; `ops['iU']` overwritten |
| Final merge | `rk:1065` → `merging_function(mode='ccg')` | | all spike trains | changes `st`, `tF`, `Wall` in place |
| Save | `rk:1097` | | | positions (needs `iU`, `iCC`, `iCC_mask`), `remove_duplicates`, PC features |

Memory [I from counts]: first-pass `tF` about 1.3 GB and matched `tF` about 2.8 GB at 20 min; about 16.6 GB at 2 h. All CPU RAM.

**Per-batch and stateless** [V]: filtering, whitening, drift interpolation, universal detection, matching, feature
computation. `run_matching` takes `(ops, X, U, ctc)` and keeps nothing between batches (`tm:270`); `Xres` is
discarded. `nm` and `B` are recomputed on every call (`tm:276, 280-281`).

**Unavoidably global** [V]: drift fingerprints, per-centre clustering over all that centre's spikes, correlogram
tests in tree and merge.

Two full universal passes run, one for drift and one for clustering (`datashift.py:198`, `rk:873`).
Round 1 clustering costs 67 s, and every later round repeats a full sweep of all 96 centres.

## 2. Spikes and templates from round 2 on (question 2)

**Universal templates are gone.** [V] `extract` receives only U (`rk:926, 383`). `run_matching` scores only U
(`tm:281`). `spikedetect` is never called on the residual. The only use of `Xres` is features (`tm:90`).

**A spike that no learned template fits** [V]:
- Peak score is `relu(B)²/nm`, thresholded at `Th_learned²` (`tm:298, 316`). So the test is
  c · ‖x‖ > Th, where c is the cosine with the best template of any unit.
- If passed, it's caught by a poor template, labelled with that id, and reaches clustering with its own waveform.
- If failed, it's missed. It stays in `Xres` unsubtracted, contaminating neighbours' features. No later step in
  that round looks for it. It can only be reached in a later round, if its unit gained a template.

**Features** [V]: `xfeat = Xres[...] @ wPCA.T` then `xfeat += amps * Ucc` (`tm:90-91`).
- `amps` is the least-squares amplitude (`tm:333`) and the subtraction was exactly `amp · U @ W` (`tm:349`).
  So add-back cancels subtraction exactly, in the PC subspace. The result is the projection of
  X minus all other peeled spikes. The template's shape does not enter.
- Confirmed in the results [V, `borrowed_templates.md`]: 12-15% of eligible units own no extraction template, and
  round 2 clustering still separates them, with cosine 0.80-0.86 to the template owner.

**What is lost for a poorly matched waveform** [V unless marked]:
1. Channels: only the 10 around the catching template's peak (`tm:45, 90`). Borrowers are usually 0 µm away,
   90th percentile 26-32 µm [V, audit].
2. Six PCs only. Same at all stages.
3. Time alignment: the time is the catching template's best lag. Spikes of one unit caught by two templates can be
   offset, with no alignment before clustering; `roll_features` runs only in the final merge (`tm:503-509`).
4. Double peels: features use the final `Xres`, so a spike peeled as `a1·T1 + a2·T2` becomes two output
   spikes whose features each lack the other's part. Rare [V, duplicates audit: 0.11-1.02 echoes per 1000 spikes].
5. Zero padding: `get_data_cpu` leaves channels outside a template's set at exact zero (`cq:636-639`), which adds
   artificial distance between spikes of one unit caught by templates with different channels.

**What limits a waveform from getting a learned template** (the stated premise). Four gates in order:

| Gate | Code | Depends on fit? |
|---|---|---|
| Score | c_u · ‖x‖ > `Th_universal` (`sd:182`). Unit-norm `wTEMP` and `weigh` (`sd:93, 258`). Rank-1: 6 shapes × 5 Gaussian sizes | Yes |
| Local maximum | largest within about 100 template positions (±120 µm on this probe) and ±20 samples (`sd:176, 181-182`), with **no subtraction** | **No** |
| Clusterable mass | enough spikes in the centre and landmarks (§3) | No |
| Separated from neighbours | graph, then tree | Only via features |

Detection is a matched filter against universal templates only: yes, in round 1 [V]. But once past the gates, the
universal template plays no part: features are raw (`sd:285-287`), and grouping uses position, not shape
(`cq:453`, `st[:,5]`).

Band of real loss [I]: a waveform with universal cosine c_u and learned cosine c_l can be missed only when
Th_l/c_l < ‖x‖ < Th_u/c_u. For H8 (7/7), c_l 0.95, c_u 0.8: 7.4 to 8.75. Larger spikes of such a unit pass, and they
are enough to birth a template, after which round 2 catches the rest. So the fit gate hides units only if
the **whole** unit lies in the band.

Measured [M]: round 1 sees 45% of round 2's spikes (kobe H8 5.32 of 11.81 M; MEC 4.89 of 11.18; kobe default 3.93 of 8.56).

| | matched in round 2 | would be hidden by local-maximum rule | left unhidden | universal actually found | share of shortfall from the rule |
|---|---|---|---|---|---|
| kobe H8 | 11.81 M | 44.2% | 6.59 M | 5.32 M | 80% |
| MEC H8 | 11.18 M | 45.8% | 6.06 M | 4.89 M | 81% |
| kobe default | 8.56 M | 38.2% | 5.29 M | 3.93 M | 71% |

Hidden fraction by score (kobe H8): 63% below 8, 58% at 8-10, 41% at 10-15, 19% at 15-25, 4.5% above 25. Hiding is
per spike, so a unit loses count rather than existence: even at score 8, 40% of its spikes get through.

Also: 53% of matched spikes (H8) have another spike within ±30 samples and 50 µm. Round 1 features have no
subtraction, so roughly half of them hold another spike's energy. [I] This, more than the universal fit, is
what later rounds fix: templates built from subtracted features, on twice as many spikes.

Caveats on this estimate: matched score stands in for the universal score; ±120 µm approximates the 100 nearest
positions; matched spikes include echoes and false positives, so the fit residual of 11% is an upper bound on that
mechanism at spike level. It says nothing direct about how many whole units are missed, which needs a raw-data probe.

**Supports the proposal:** subtraction during learning would unmask spikes and clean features. **Cuts against it:**
this doesn't need fine interleaving, since one learned round already provides it, and round-3 spike gain is only 1.5%.

## 3. What clustering needs (question 3)

Per centre: centres are every 40 µm with one x-centre here, 96 in all (`cq:421`; logs). Each spike belongs to the
centre of its position or template, and no clustering crosses centres. Only the final merge can join across them.

| Part | Code | Global need | On a small buffer? |
|---|---|---|---|
| Spikes per centre | `cq:496-511` | `ix[PID]` scans all spikes per centre | yes |
| Group floor | `cq:522`, `cluster_min_group_spikes` 1000 | absolute count; below it, one cluster | yes if ≥ 1000 in the buffer |
| Landmarks | `cq:28-35`, every `nskip`-th spike, cap 25000 | `nskip` absolute | yes, with `nskip` lowered |
| kNN | `cq:49-51`, 10 neighbours (H8 the same) | none | yes |
| kmeans++ | `cq:210`, 200 seeds | needs ≥ 100 spikes (`topk` of `ntry`, `cq:207`) | yes; cost floor (§7) |
| Graph | `cq:174-185`; resolution term depends on cluster's share of graph, not absolute size [I] | none | yes |
| Tree | `hierarchical.py` | none | yes |
| Refractory test (template pass only) | `ss:139-143`, `cq:525-528` | needs spike pairs | **no** |
| Bimodality | `ss:40-51`, 400 bins, smoothed σ 4 bins | hundreds of spikes [I]; off in H8 | weak |
| Template | `cq:567-569` mean of Xd | none | yes |
| 'mu' merge | `tm:421`, sizes only | none | yes |
| 'ccg' merge | `CCG.py:91-92`, needs Q12 < .05 | needs spike pairs | **no** |

There is no per-cluster minimum size in KS4 [V, grep]; the floors are the group floor, `CCG.refract`'s `> 10`
(`CCG.py:130`), and the implicit landmark floor.

**H8's first-pass tree does almost nothing** [V]. `meta=None` in spikes mode, and with bimodality 0 a node is kept apart
unless the score is negative, a rare edge case (`ss:50, 156`). So round 1's clusters are the graph clusters: 3933.
Then 'mu' merge reduces these to about 2450. H8's template-pass tree merges only by refractoriness, so at H8 **the tree's
one merging test is the one that needs the most data.**

**The sparse shortcut is a trap on short buffers** [V]. In the tree, `sparse = Q00 < .25 and not require_significance`, and
`cross_refractory = R12<.25 and (Q12<.05 or sparse)` (`ss:70-72`). Sparse pairs usually have R12 of 0, so
they merge untested [I]. At full length it decided only 4 of 157 merges [V, `search_round2.md`]. Q00 is about
r1·r2·T·0.001, so a pair stops being sparse at T ≈ 250/(r1·r2) s. Short buffers would flip the tree to
merging low-rate pairs by default. `tree_refractory_significance=True` would be needed for learning rounds. The final merge has
the opposite bias: no shortcut (`CCG.py:91-92`), and p < .05 needs Q00 ≥ about 0.27 even at zero coincidences [I from
`CCG.py:64-68`], so low-count pairs don't merge. So the two tests disagree in direction when counts fall.

T in correlograms cancels in R12 (`CCG.py:45-47, 60, 75`) [V]: a gappy subsample doesn't bias it.
I expected that to be a trap for strided batches. It isn't. Only counts fall.

**Minimum batches at 30 kHz, 2 s batches.** A 5 Hz unit gives 10 matched spikes per batch.

| Need | Requirement | 5 Hz unit, batches |
|---|---|---|
| Centre group ≥ 1000, template pass | 196 spikes/batch per centre (p10 109) | 6 (p10 10) |
| Centre group ≥ 1000, round 1, H8 | 87 per batch (p10 52) | 12 (p10 20) |
| Landmarks: ≥ 11 so a spike's neighbours can all be its own (self excluded, `cq:64`), H8 `nskip` 10 | 110 spikes | **11** (25 in round 1 at 45% detection) |
| Same, default `nskip` 20 | 220 spikes | **22** (50 in round 1) |
| Same, `nskip` 1 | 11 spikes | 2 |
| Non-sparse correlogram, 5 × 5 Hz | T ≥ 10 s | 5 |
| Non-sparse correlogram, 1 × 1 Hz | T ≥ 250 s | **125** |

These are floors for a neighbourhood to exist, not for clusters to be good. [I] A working margin is 3-5× more.

But 5 Hz is above typical: the median unit here is 2.4 Hz (kobe round 2), lower quartile 1.2, 10th percentile 0.46.
Batches to reach 110 matched spikes: 5, 23, 118 at 10th/50th/90th percentile of units; 2.2× more in round 1. So:

- **Could run per batch:** matching, features, mean update of existing templates.
- **Could run on a rolling buffer of tens of batches:** graph clustering, tree without correlograms, 'mu' merge.
- **Need many spikes per cluster:** refractory tree merges, final merge, and clusters of low-rate units.

Template averaging is a different matter from clustering. [I] Ten spikes give a mean with PC-space noise of about
√(60/10) ≈ 2.4 against a norm near 10 (unit-variance whitened noise assumed), so a cosine near 0.97. Templates can be
refined per batch **if assignment is right.** What needs mass is deciding the groups, not averaging them.

## 4. Templates and the assumption of a fixed dictionary (question 4)

**How Wall and U are built** [V]: `W[j, ichan, :]` = cluster mean of Xd (`cq:566-569`), all other channels zero.
`align_U` rolls each template so its best `wTEMP` lag is centred (`tm:147-159`). 'mu' merge joins pairs with correlation
> threshold and norms within 20% (`tm:497-498`), weighted by spike count (`tm:511`). Amplitude is kept, so U is in the
same units as `tF`: a batch mean of `tF` for a template is directly usable.

Note a template's support is `ichan` (12-15 channels) but a spike's features span 10. A mean update from `tF` covers
only those 10, so the rest need holding or a raw-data read.

**Computed once per dictionary** [V]: `iCC`, `iCC_mask`, `iU`, `Ucc` (`tm:59-64`); `ctc` (`tm:68`).
- `iCC` depends only on geometry. `iU`, `Ucc` depend on U.
- `ctc` is an immutable CSR-like dict. Insert means rebuild; [M] that takes 0.024 s at 1761 templates, 5.6% non-zero.
  About 0.05 s at 2450 by n² [I]. CSR assembly isn't in the timing but is a sort and cumsum. Rebuilding every batch
  adds about 15%. Rows for 256 changed templates took 0.004 s.
- `_peel_fused` has no capacity tied to template count: `flags`, `peaks` by NT; `kmax` from `ctc` (`tm:374-378`, `cuda_kernels.py:135`).

**So matching itself assumes nothing about a fixed dictionary.** Replacing U and `ctc` between batches needs only moving
`tm:59-68` into the loop.

**What does assume it** [V]:

| Assumption | Where | What breaks |
|---|---|---|
| `st[:,1]` alone records which channels a spike's features are on, via `iCC[:, iU[id]]` | `tm:90`, `cq:347-348`, `postprocessing.py:47, 54` | A template whose peak moves relabels the channels of old spikes. Needs frozen `iU` per id, or per-spike channels |
| Template ids are permanent indices into one U | `cq:449, 473`, `get_data_cpu`, `make_pc_features` | Deaths and merges need a stable id table |
| Centres are assigned from `iU` | `cq:347-351, 473` | A moved peak changes group |
| `ops['iU']` matches the latest extraction | `tm:64` | Final extraction must come last. D2 and D3 meet this |
| `augment_templates` keeps ids | `tm:197-200` | Only needed for that mode |
| `wPCA` and `wTEMP` fixed | `tm:90, 148-158`, `sd:149` | Recomputed by every `sd.run` (`sd:220-227`), so a later `sd.run` would invalidate U |
| `align_U` per dictionary | `tm:163` | Online updates need re-alignment, or spike time shifts |
| `dshift` by batch | `io.py:1137` | See D3 trap |

**Not assumed** [V]:
- Batch order: `st` sorted at the end (`tm:127-143`), `t_shift` from `ibatch`, only `ibatch == 0` is special (`tm:93`).
  Random order costs only prefetch, since `_BatchReader` guesses forward steps (`io.py:636`).
- Spike time dedupe: `remove_duplicates` runs at save, by cluster (`io.py:363`). Batches don't double-report, as
  `Cf` is zeroed in the padding (`tm:299-300`).
- Drift: the frame is fixed before any template exists.
- Old features remain valid after a dictionary change, since they are the waveform, not a template. Only their subtraction
  quality is out of date. [I] So a rolling feature buffer doesn't need re-extraction.

## 5. Insertion seams (question 5)

Shared prerequisites [V]: all clustering needs `ycup`, `xcup`, `dmin`, `dminx` (`cq:416-417, 450, 456`), `iC` for spikes mode. With
`nblocks=0` these exist only via `sd.run`. Any design that drops round 1 must still run `sd:216-258`.

Bit-identity off: all designs gate behind a default-off setting as `extra_matching_rounds` does (`rk:317-345`), checked
by `tuning/same_sort.py`. Refactors of `extract` must be pure code motion.

Seeds [V]: `kmeans_plusplus` reseeds torch and numpy at every centre (`cq:228-229`), so clustering is insensitive to
earlier RNG use. But `sd.run` isn't: `TruncatedSVD` and `KMeans` take no `random_state` (`sd:78, 91`) so use numpy's
global RNG [I]. New code before `rk:873` must use its own `Generator`. `_sort`'s seed block sits before drift.

Stage cache: new settings aren't in `LATE`, so they fall into key A and just miss (`stage_cache.py:70`). Safe, no reuse. A sort skipping
round 1 skips cached stage B, which also holds round 2's extract. New sorts cost full price.

### D1: full online learner replacing the pre-final clustering

Seam: new `online_learn(ops, bfile, device) -> U` replacing `rk:323`. Then `_extract_learned` (`rk:934`), `_cluster_templates`, `_final_merge`.

Must change:
- split `sd.run` into setup (`sd:216-258`) and loop; keep second-pass `wPCA` recompute (`sd:220-227`) on corrected data;
- split `extract`'s batch body (`tm:88-113`) into a function, moving `tm:59-68` under a dictionary-version check;
- new: births from `template_match(Xres, ops, iC, iC2, weigh)`. Its signature takes any X (`sd:140`), so `Xres` works directly;
- new: per-centre buffers carrying channels, since residual-born spikes are on position channels, not template channels;
- new: birth rule, update rule, deaths, merging, id table, bootstrap. `run_matching` fails on an empty U (`tm:301`, `torch.max` over dim 0 of zero rows [I]);
- diagnostics: `Wall0`, `clu0` for `plot_diagnostics` (`rk:338`) and GUI.

Stays: drift, final extraction, final clustering, merge, save.

Big problem [I]: per-batch births can't use KS4's clustering (§3), so births would be KS2.5-style from one or a few residual
spikes, or per-centre on a buffer. The first discards what makes KS4 good (audit §1: gains are in steps 3/5/6, none imported
from KS2.5 helped). The second is rounds in disguise: a 5000-spike buffer is 25 batches and a sweep costs 8.6 s to
extraction's 4.3 s. New untuned parameters too: H8 was tuned for this pipeline, and interactions are strong (audit §5).

Invariants: off stays identical. Determinism fine with a private generator. Every downstream id invariant is preserved
because a final `extract` restarts from a fixed U. Largest cost of the designs.

### D2: online warm-up seeding the dictionary, then the unchanged rest

Seam: inside `detect_spikes`, replace or feed `rk:873-907` with a warm-up producing `Wall3`, keep `rk:926`.

Variants:
- **D2a** replaces round 1: same new code as D1, with less at stake.
- **D2b** adds refinement only, no births: after `rk:905`, run a few hundred batches of matching and update means. No clustering needed, so per-batch is
  feasible. Uses `tm:88-91` as is, plus re-align and `ctc` rebuild. Limitation: the dictionary doesn't shrink, and results say
  rebuilding from new clusters (which shrinks it 25-30%) is what mattered: augment gained half, and nothing on the
  curves. [I] Refinement alone may not capture the gain. Cheap to test.

Stays: all of `_cluster_templates` onward. Bit-identity gate at `rk:873`. Replacing round 1 alone saves 100 s (33 + 67).

### D3: ordinary rounds on a subsample of batches, then one full final extraction

Seam: `_sort` only. After `rk:289`, build a second `BinaryFiltered` for learning, with `batch_downsampling=k` and
`dshift=ops['dshift'][::k]`. Pass it to `detect_spikes` (`rk:323`) and `_extract_learned` in non-final rounds (`rk:383`). Final round uses `bfile`.

**The trap** [V]: `BinaryRWFile.padded_batch_to_torch` multiplies a local `ibatch` by k (`io.py:893`), but
`BinaryFiltered` passes the original index to `filter` (`io.py:1159-1160`), which reads `dshift[ibatch]` (`io.py:1137`). Under global `batch_downsampling` this
is consistent, because drift was computed on strided batches. Changing the stride after drift isn't. On these recordings the
error would be small and easy to miss. Time is right either way (`tm:103`, `sd:289`). `io.py:936` compares strided and
unstrided indices, harmless [V: the branch is then a no-op for full-size data].

Must also decide:
- `cluster_downsampling` scaled by f: keeps landmark count exactly, and with it the unit floor (§3). At f = 0.2, `nskip` 10 → 2;
- `cluster_min_group_spikes` scaled; no group near 1000 now at p10 31k, so low risk until f ≈ 0.03;
- `tree_refractory_significance=True` in learning rounds, or low-rate pairs merge by shortcut;
- `wPCA`: `sd.run` on the sub-file uses k× fewer clips (`sd:57`). Better computed on the full file;
- settings per round: `cq.run` reads `ops['settings']` at entry (`cq:458-469`), so an `ops_k` copy as at `rk:362` works.

Stays: everything else. Off: same `bfile`, identical. Seeds: unaffected.

Coverage (kobe round 2 units; unchanged `nskip`):

| f | units ≥ 110 spikes | their share of spikes lost if below | with `nskip` scaled |
|---|---|---|---|
| 1 | 98.6% | 0.01% | same |
| 0.5 | 96.7% | 0.06% | same as f = 1 |
| 0.2 | 90.2% | 0.45% | same as f = 1 |
| 0.1 | 80.1% | 1.8% | same as f = 1, to `nskip` 1 |
| 0.05 | 61.6% | 6.6% | below `nskip` 1: floor rises |

MEC is similar. Round 1 counts are lower, more so for small units. A missed template is not fatal: a final-round spike can be
caught by a borrowed one, and final clustering may still separate it, as orphans show.

Time [I, assumes linear]: f = 0.2 takes learning from 563 to about 113 s, total 826 → about 376 s, 2.2×.

Evidence for: KS4 changes ≤ 1.4% from 20-min to 2-h sorts, and the 2-hour and 20-min curves of each round nearly
coincide (`round3_results.md` §4). So at 2 h, f = 1/6 is already a data amount known to work, though there it's contiguous, not strided.
Evidence against: 2-hour cost isn't exactly linear; `cluster_downsampling` matters in tuning, so `nskip` scaling isn't neutral; f is untested.

### Others the code suggests

**D3b: rolling rounds.** Each round uses a different stride phase, so the subsample isn't reused and the rounds approach a full pass
overall. This is the proposal's idea with existing machinery. It needs an offset in `padded_batch_to_torch` and at `tm:103`, `sd:289`,
or a batch list argument. Reasonable next step after D3. The cost per sweep matches D1's rolling buffer (§7), so D1 has no cost advantage here.

**D4: residual rescue.** In the extract loop, run `template_match(Xres, ...)` after `tm:89` and keep hits. Directly targets spikes that no
template reached. Same universal gate, though, so it recovers masked spikes rather than poorly fitted ones. Their features are on position channels, so they need
spikes-mode clustering, then `augment_templates`. Adds about 0.05 s per batch. Results argue for caution: augment gained less than repeat.

**D5: warm-started clustering.** `cluster()` already accepts `iclust` (`cq:125, 167-172`), but `cq:531` doesn't pass it. Seed each centre from the
template ids of its spikes. Carries state between rounds, which at present survives only through U. May cut churn. Risk: graph clustering
only merges and reassigns seeds, so it can't split a seed; would need added kmeans++ seeds. Note `kn` passed without `M` would crash
(`cq:138-143`); use `iclust` alone. Changes `niter` exit time (`cq:183`).

**D6: shrink the local-maximum neighbourhood.** `nearest_templates` (default 100) sets its spatial reach (`sd:252`). It goes straight at finding 3: a
smaller neighbourhood gives more first-pass spikes with no new algorithm. It appears in none of the tuning documents, so it hasn't been searched [V, grep].
The time window `nt0min` also sets trough alignment (`rk:567`, `tm:106`), so it isn't a free knob. Whether more first-pass spikes help is unknown, since the extra ones are the overlapped ones.

### Comparison

| | New code | Touches ids/channels | Keeps KS4 clustering | Speed-up (20 min, 4 rounds) | Main risk |
|---|---|---|---|---|---|
| D1 | large | yes, during learning | final only | ≤ 3× [I] | loses what makes KS4 good |
| D2a | large | during warm-up | from round 2 | 1.1× if it only replaces round 1, about 1.6× if it also saves a round [I] | same, smaller |
| D2b | small | no | yes | none or slower alone | may not capture gain |
| D3 | small | no | yes | about 2.2× at f = 0.2 [I] | rare units; drift trap |
| D3b | small | no | yes | same | as D3 |
| D5 | small | no | yes | a little, in `niter` | freezes errors |

## 6. Summary: for and against

**For interleaving:**
- Matching accepts a changing dictionary at low cost; `ctc` is not a barrier.
- Features survive dictionary change, so buffers don't need re-extraction.
- Means can be updated from 10 spikes if assignment is right.
- Rounds do help, and first-pass spikes are heavily masked and overlapped, so earlier subtraction has real targets.
- Four rounds aren't at a fixed point: units change, and reproducibility is still rising.

**Against fine interleaving, at batch level:**
- Speed cap about 3×, and D3 reaches 2.2× with little code.
- The clustering can't work below tens to hundreds of batches, and costs 50× a batch of matching per sweep.
- At H8 the tree's only merge test is correlogram-based, and is misleading on short buffers.
- The gain so far is cleaning, not discovery, and comes with dictionary rebuilds from global clusters.
- The transfer record from KS2.5 is poor (audit §5).
- Effects interact, and H8 was tuned with two rounds.

Not established here: whether any of this improves quality per pass. Nothing was run.

## 7. Measurements

Scripts are in the session scratchpad, not kept. Nothing was written to the worktree, and no sort was run.

- **Stage times** from `C:\data\ks4_tuning\exploratory\iteration\sorts\{kobe_rounds0_s1, kobe_repeat_s1, kobe_2h_repeat_s1}\kilosort4.log`. 20 min H8:
  drift 32, universal 33, first clustering 67, extract 132, cluster 114, extract 102, cluster 115, extract 101, cluster 118, merge 7, save 7 s.
  `kobe_repeat_s1` used the stage cache, so its early stages are from `kobe_rounds0_s1`. 2 h total 6005 s: drift 220, universal 174, cluster 498, then
  872 + 924, 660 + 902, 650 + 926, merge 64, save about 49.
- **Spikes per centre** from `Center | Xd shape` lines in those logs. Template pass, kobe H8: median 118k, p10 66k. First pass: 52k, p10 31k.
- **Unit rates** from `spike_clusters.npy`, 1200 s, 1865 units, round 2.
- **`ctc` timing:** `prepare_matching`'s einsums on `Wall.npy` of `kobe_repeat_s1`, 1761 × 374 × 6, RTX 5070 Ti, GPU idle. Excludes the final sort into CSR. Final
  clusters stand in for U, so sparsity is approximate.
- **`B` per batch:** 0.038 s on random data, at 1761 templates.
- **Clustering timing:** `cq.cluster` + `maketree` from the worktree, on synthetic 20-Gaussian mixtures, 72 dims, resolution 2:
  0.085 s at 1000 spikes, 0.090 at 2000, 0.09-0.10 at 5000, 0.14-0.31 at 20000, 0.77 at 118000. Real per-centre average from logs is 1.2 s at 118k,
  including `get_data_cpu` and the split. Synthetic data probably understates real kNN and tree cost.
- **Local-maximum estimate:** on saved round-2 sorts (spike time, y, `score.npy`), a spike counts as hidden if a higher-scored one lies within ±20 samples and ±120 µm.
