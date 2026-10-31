# KS4 settings search: harness

Searches Kilosort 4 settings for rat 26648 ("kobe"), 2020-08-21, on the 20-minute excerpt 1668–2868 s.
The rules are in [evaluation_spec.md](evaluation_spec.md); this file says where things are and how to
run them.

## Where things live

**Code and spec** (this folder, branch `ks4-tuning`):

| File | Role |
|---|---|
| `evaluation_spec.md` | The spec: score, cleaning, guards, Phase 0, search rules |
| `config.json` | Paths and fixed parameters used by the wrapper |
| `run_candidate.py` | Wrapper: `freeze`, and `run` (sort → clean → evaluate → compare → log) |
| `run_sort.py` | One KS4 sort from this checkout, with `provenance.json` |
| `cleaning.py` | Multi-unit events, duplicate test, masks (§6.1–6.3) |
| `waveform.py` | Bombcell noise screen and the stricter G5 screen (§6.4) |
| `score.py` | S, S_track, C, eligibility (§5, §7.2, §7.5) |
| `evaluate.py` | Per-sort evaluation (`evaluation.json`) |
| `compare.py` | Guards against default replicates, matched units (§7) |
| `hybrid.py`, `make_hybrid.py` | Hybrid ground truth: scoring (G4) and construction (§7.4) |
| `theta_reference.py`, `build_reference.py` | Theta cycles and the frozen reference heading θ_ref (§5.1) |
| `acg_class.py` | Theta-source units (elv environment) |
| `extract_segment.py`, `extract_tracking.py`, `make_probe.py` | One-off extraction of the excerpt, tracking and probe |
| `raw_artefacts.py`, `plot_raw_events.py` | Raw transient diagnostic and safety-net mask (§6.1) |
| `replicates.py`, `alpha_scan.py`, `self_leak.py`, `perturb.py`, `adjudicate.py`, `reference_heldout.py`, `reference_invariance.py`, `compare_references.py` | Phase 0 checks (§8) |
| `ring_reference.py` | Superseded HMM reference (kept for the record) |

**Data and results** (`C:\data\ks4_tuning\26648_kobe_2020-08-21\`):

| Path | Contents |
|---|---|
| `search\ap_1668-2868s.bin` | The excerpt (27.7 GB, verified against the source) and its manifest |
| `search\hybrid\` | Hybrid excerpt (the search input), `hybrid_truth.npz`, `hybrid.json` |
| `search\tracking_1668-2868s.npz`, `search\probe_3a_opt3.json` | Tracking and probe |
| `search\raw_artefacts_car.npz` | Raw transient candidates after referencing (the mask's source) |
| `search\figures\` | Diagnostic figures |
| `runs\reference_seed1000\` | Reference sort and the frozen `theta_ref.npz` (second reference: `reference_seed1001`) |
| `runs\hybrid_default_seed1..5\` | Default replicates on the hybrid excerpt (the comparison baseline) |
| `runs\phase0_*.json` | Phase 0 results |
| `runs\frozen.json` | Hashes recorded at freezing |
| `runs\search_log.jsonl` | Append-only search log (one `start` and one `result` record per run) |
| `runs\search\<run_id>\` | Each search run: `sort\` (KS4 output, evaluation), `comparison.json`, `summary.json`, logs |
| `runs\test_log.jsonl`, `runs\search\test_*` | Wrapper tests, outside the search log |

## Environments

- `kilosort` (editable install of this checkout; refuses any other copy): everything except `acg_class.py`.
- `elv`: `acg_class.py` only.
- `npx-load`: `extract_tracking.py` only.

## Running the search

1. Freeze once, after committing: `python run_candidate.py freeze`. This hashes the harness and the
   frozen inputs and writes the first log record. Later changes to anything hashed make `run` refuse
   to score until the spec gets a new version and the harness is refrozen.
2. Each evaluation:
   `python run_candidate.py run --name <short_name> --settings '<json>' --seed 1 --rationale "<one line>"`
   Phases: `search` (default), `replicate` (seeds 2, 3 of a promising run), `ablation`, `test`.
3. The optimising agent reads only `summary.json` and the log's `result` records (§3). Per-unit
   scores are in each sort's `evaluation.json` and are not read during the search.

A run takes about 7 minutes of GPU for the sort and under a minute for the harness. 100 runs, of
which 15 are reserved for replication and ablation (§9).
