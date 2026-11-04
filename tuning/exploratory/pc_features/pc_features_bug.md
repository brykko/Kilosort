# `make_pc_features` writes most spikes' features on the wrong channels

**Confirmed, on data.** In stock KS4, `pc_features.npy` rows mostly do not match `pc_feature_ind.npy`
for the spike's own unit. On five saved sorts, 82–85% of spikes (default parameters) and 95–96% (H8)
have at least one wrong slot. 39–53% have the wrong data in slot 0, the unit's peak-channel column
(table below).
With the fix, every row is correct. The harness's scores don't read the saved PC features, so past
results don't change. What is affected is Phy (FeatureView, and Phy's own feature-based splitting
and similarity) and anything that reads KS4's `pc_features.npy`, such as a KS2-style splitter fed
KS4 features. Our tuning sorts didn't save PC features, so there is no damaged file of ours on disk.

Upstream: [MouseLand/Kilosort#1039](https://github.com/MouseLand/Kilosort/issues/1039) (opened
2026-07-26, still open) describes the same mechanism on a four-spike toy. A collaborator replied
(2026-08-14) that final clusters sharing a template "should never occur" and asked for real data.
`upstream/main` (0e07514, 2026-09-25) has the same logic. Our fork's version (5e05b53, which uses
precomputed index groups) selects the same `igood` as upstream.

## The mechanism (`kilosort/postprocessing.py:115-141`)

For each final cluster `i`, in ascending id order:

1. `iunq` = the detection templates of cluster `i`'s spikes.
2. `igood` = **every** spike detected with those templates, including spikes now in other clusters.
3. `get_data_cpu` reads `tF[igood]` and puts each spike's `nearest_chans` columns onto the union of
   the templates' channels. It assumes each row is still in its template's `iC` channel order.
4. The channels are ranked by the norm of the mean feature, and `tF[igood]` is **overwritten** with the
   top `nearest_chans` columns in cluster `i`'s order. `feature_ind[i]` records that order.

A spike whose template also feeds another cluster is therefore written once per such cluster. From
the second write on, step 3 reads a row that is already in another cluster's channel order as if it
were in template order, so its values land on the wrong channels or are dropped. The last write uses
the last cluster's channel list, which is usually not the spike's own. Rows are only correct when a
template's spikes all end up in one cluster.

That case is rare in KS4. The final clustering (`clustering_qr.run`, `mode='template'`) uses template
ids only to group spikes by position and to lay out their features. It clusters spikes one at a time,
so one template's spikes often go to several units. That answers the doubt in #1039. With default
parameters, about 90% of detection templates that have spikes send them to more than one final unit,
and 86–88% of units mix templates. With H8 the figures are 98% and 94–95%.

## Measurement (`pc_features_audit.py`)

No `tF` was saved, so the script builds a synthetic one from each sort's saved
`spike_detection_templates`, `spike_clusters`, `ops` and `templates.npy`. Each row slot holds two
"PCs":

- PC0 is the template's amplitude on that channel, so the channel ranking is realistic. `templates.npy`
  is per final unit, so a detection template's amplitude is the spike-weighted mean over the units its
  spikes went to.
- PC1 is a channel label (`(id + 1) · 1e-3`), small enough that it doesn't change the ranking.

The script runs stock `make_pc_features` and the fixed version on this tensor, CPU only. A slot is
wrong when its label differs from what `pc_feature_ind[own unit, k]` promises: that channel if it is in
the spike's template neighbourhood, else zero padding. Outputs are in
`C:\data\ks4_tuning\exploratory\pc_features\` (`pc_features_audit.json`, `.log`).

| Sort | spikes | units | units mixing templates | written > once | last writer ≠ own | rows wrong (stock / fixed) | slot 0 wrong (stock) | own data zeroed (stock) | slot 0 = peak (stock / fixed) |
|---|---|---|---|---|---|---|---|---|---|
| roger H8_s1 | 11.18 M | 2133 | 94.7% | 95.7% | 64.8% | 95.7% / 0.0% | 50.9% | 17.0% | 51.8% / 91.4% |
| roger H8_s2 | 11.17 M | 2174 | 94.1% | 95.0% | 64.0% | 95.0% / 0.0% | 52.9% | 17.0% | 53.2% / 93.1% |
| roger default_s1 | 8.12 M | 846 | 86.4% | 84.7% | 53.3% | 84.6% / 0.0% | 41.7% | 8.9% | 57.2% / 94.0% |
| roger default_s2 | 8.07 M | 821 | 86.5% | 84.4% | 52.0% | 84.0% / 0.0% | 42.5% | 8.7% | 56.5% / 94.3% |
| kobe nohybrid_default_seed1 | 8.56 M | 804 | 88.4% | 81.8% | 50.7% | 81.6% / 0.0% | 38.6% | 8.6% | 59.6% / 95.3% |

Each spike is written 3.3–3.5 times on average with default parameters and 7.7–8.0 times with H8.
Of the detection templates that have spikes, 89–91% (default) and 98% (H8) send them to more than one
unit.

Column meanings:

- *written > once*: share of spikes whose template feeds more than one unit.
- *last writer ≠ own*: share of spikes whose row was last written by another unit.
- *rows wrong*: share of spikes with at least one wrong slot.
- *slot 0 wrong*: share of spikes whose peak-channel column holds the wrong data.
- *own data zeroed*: share of slots where the spike has data on that channel but the row shows zero.
  This is #1039's "zero-valued PCs on peak channels".
- *slot 0 = peak*: share of units whose `pc_feature_ind[:, 0]` is the argmax channel of the unit's
  template.

*Rows wrong* is within 0.4 points of *written > once*: nearly every spike that is written more than
once ends up corrupted (a few come out right by coincidence), and no spike written once does. The
stock `pc_feature_ind` is also off, because the ranking pools other units' spikes, some of whose rows
are already scrambled. Its slot 0 is the unit's peak channel for 52–60% of units, against 91–95% with
the fix. The fix's shortfall from 100% was not looked into. A likely contributor is the approximation
of detection-template amplitudes from per-unit templates.

## Minimal fix

Restrict `igood` to cluster `i`'s own spikes. Each row is then read once, in template order, and
written once, in its own cluster's order. The channel ranking uses only the cluster's own spikes. In
`make_pc_features` (`kilosort/postprocessing.py`):

```python
    clu_order, clu_bounds = groups(spike_clusters)
    pid = torch.from_numpy(spike_templates).long()

    for i in np.unique(spike_clusters):
        # Spikes of cluster i (ascending), and the templates they came from
        igood = clu_order[clu_bounds[i]:clu_bounds[i+1]]
        iunq = np.unique(spike_templates[igood]).astype(int)
        ix = torch.from_numpy(np.zeros(n_templates, bool))
        ix[iunq] = True
        Xd, igood, ichan = get_data_cpu(
            ops, xy, iC, pid, tF, None, None,
            dmin=ops['dmin'], dminx=ops['dminx'], ix=ix, merge_dim=False,
            igood=torch.from_numpy(igood)
            )
        ...  # unchanged from here
```

`tmp_order`/`tmp_bounds` are then unused. When every template belongs to one cluster, the output is
bit-identical to stock (tested). It should also be faster, because each spike is processed once
instead of 3.3–8.0 times on average (not timed). The same fix suits upstream (`igood=` is already a parameter
of `get_data_cpu` in our fork; upstream would pass `(spike_clusters == i).nonzero()`). The fix is not
applied to `kilosort/` here.

## Test (`test_pc_features.py`)

The test uses a toy probe (8 channels, one template per channel, 3 neighbours) with random `tF`. It
checks every output value against the spike's original features placed on its own unit's
`pc_feature_ind` channels, with zeros elsewhere.

- `test_fixed_rows_match_own_cluster`: the fix is exact when templates are split between clusters.
- `test_stock_rows_wrong_when_templates_shared`: stock fails the same check, which documents the bug.
- `test_fixed_equals_stock_without_sharing`: with one cluster per template, stock is correct and the
  fix is bit-identical to it.

pytest isn't in the `kilosort` environment, so the test also runs as a script: `python
tuning\exploratory\pc_features\test_pc_features.py`. All three tests pass.

## Not done

- No real `tF`, so the size of the error in feature space (as opposed to which slots are wrong) isn't
  measured. A sort with `save_extra_vars=True` would allow that, but it needs a GPU.
- The fix isn't applied to `kilosort/postprocessing.py`, and nothing has been posted on #1039. Our
  numbers would answer the collaborator's question there.
