# Room for a third extraction round (2026-11-03)

KS4 extracts spikes with the templates of its first clustering, then clusters them again. A third round
would extract with templates of the final clusters, and can only help final units that the first clustering
didn't give a template of their own. `borrowed_templates.py` measures how many such units there are and
what borrowing costs them, read from the scored sorts of `autocurate\sorts\` (seed 1; kobe is the hybrid
excerpt, with hybrid copies excluded). Outputs: `C:\data\ks4_tuning\exploratory\iteration\`.

Each extraction template's *owner* is the final unit that got most of its spikes. A unit's spike is
*borrowed* if it was caught by a template another unit owns; an *orphan* owns no template. *Fit* is
score / amplitude (the matching score over the norm of the spike's features on the template's channels),
the cosine between the spike and the template that caught it.

| | kobe H8 | kobe default | MEC H8 | MEC default |
|---|---|---|---|---|
| Extraction templates / final units | 2411 / 1904 | 936 / 849 | 2526 / 2133 | 903 / 846 |
| Eligible units that are orphans | 13.1% | 12.1% | 15.2% | 13.3% |
| Eligible units with ≥ 50% borrowed spikes | 16.4% | 13.1% | 18.6% | 13.9% |
| ... their share of spikes | 4.7% | 2.2% | 6.7% | 2.6% |
| ... their share of S | 5.2% | 5.8% | | |
| Fit of borrowed against owned spikes, same unit and amplitude (median) | 0.949 | 0.928 | 0.954 | 0.930 |
| Extra spikes from a template of their own, ≥ 50%-borrowed units (rough) | +9.2% | +6.6% | +2.7% | +3.7% |
| ... as a share of all spikes | +0.43% | +0.15% | +0.18% | +0.09% |
| Same estimate in units that own their templates (control, should be ~0) | −1.0% | −0.5% | −1.5% | −1.0% |
| Runtime a third round adds (extract + cluster) | 242 of 386 s | 125 of 230 s | 248 of 389 s | 124 of 229 s |

- **The gap is real and common.** In every sort 12-15% of eligible units own no extraction template.
  Their spikes come from a template peaked on the same channel (median distance 0 µm, 90th percentile
  26-32 µm), owned by a unit of a different waveform (median cosine between the two units' final templates
  0.80-0.86). These are units the first clustering lumped with a neighbour at the same site and the final
  clustering split apart. Only 13-53 per sort look like near-copies of their owner (cosine ≥ 0.95).
- **Borrowing costs little.** At the same amplitude, borrowed spikes fit their template about 5% (H8) to
  7% (default) worse than the same unit's owned spikes. The shared template was learned from both units'
  spikes, so it fits the borrower better than the owner's final waveform would suggest.
- **So a third round's recall gain would be small:** a few percent more spikes in the units it acts on,
  0.1-0.4% of all spikes. On kobe H8 those units carry 5.2% of S (~73 bits/s); a gain in proportion to
  their spikes would be ~7 bits/s, a quarter of δ (28.6).
- **Not measured:** effects of a third round beyond recall: better subtraction of these units' spikes
  (fewer echoes and cross-unit re-fits), cleaner features for overlapping neighbours, regrouping. Only a
  run would show them.
- **Correction to an earlier claim:** extraction templates outnumber final units only 1.1-1.3×, so seeding
  a third round with the merged units' templates would not collapse an over-complete dictionary. The
  duplicates audit's "median of 9 extraction templates per final unit" counts the templates a unit's spikes
  came from, not templates per unit.

Caveats: one seed per sort. The extra-spike count assumes the score density just below `Th_learned` equals
the density in [Th, 1.25 Th], and is calibrated by owned units of similar amplitude (raw, uncalibrated, it
gives +5-7% even to owned units, from differences between units unrelated to their templates).
