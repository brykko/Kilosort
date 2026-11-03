#!/bin/bash
# Scoring of make_variants.py's sorts with the harness's own tools, outside run_candidate.py (nothing is logged
# to the search log), one job at a time. The uncut seed-1 reruns are scored the same way, for the baseline.
# Kobe: evaluate.py (config.json's reference, t0, duration, hybrid truth), then compare.py against the five
# default replicates. MEC: transfer.py at the transfer test's alpha (0.0625 s) against the three default sorts.
set -e
cd /c/code/github/Kilosort/tuning
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
K=C:/data/ks4_tuning/26648_kobe_2020-08-21
M=C:/data/ks4_tuning/25843_roger_2019-09-11
A=C:/data/ks4_tuning/exploratory/autocurate
V=$A/variants
DEF_K="$K/runs/hybrid_default_seed1 $K/runs/hybrid_default_seed2 $K/runs/hybrid_default_seed3 $K/runs/hybrid_default_seed4 $K/runs/hybrid_default_seed5"

for s in $A/sorts/kobe_H8_s1 $A/sorts/kobe_default_s1 $V/kobe_*; do
  [ -d "$s" ] || continue
  echo "=== kobe $(basename $s) $(date +%T)"
  $PY evaluate.py --sort "$s" --reference "$K/runs/reference_seed1000/theta_ref.npz" --t0 1668 --duration 1200 \
    --hybrid-truth "$K/search/hybrid/hybrid_truth.npz" --raw-artefacts "$K/search/raw_artefacts_car.npz" > "$s/evaluate.log" 2>&1
  $PY compare.py --candidate "$s" --default $DEF_K --raw-artefacts "$K/search/raw_artefacts_car.npz" --out "$s/comparison.json" > "$s/compare.log" 2>&1
done

echo "=== mec $(date +%T)"
$PY transfer.py --default "$M/runs/default_s1" "$M/runs/default_s2" "$M/runs/default_s3" \
  --winner $A/sorts/mec_H8_s1 $A/sorts/mec_default_s1 $(ls -d $V/mec_*) --alpha 0.0625 \
  --tracking "$M/search/tracking_13575-14775s.npz" --raw-artefacts "$M/search/raw_artefacts_car.npz" --t0 13575 --duration 1200 \
  --out "$V/transfer_mec.json"
echo "=== done $(date +%T)"
