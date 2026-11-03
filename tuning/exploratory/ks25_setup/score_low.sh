#!/bin/bash
# Scoring of the KS2.5 "standard_low" sorts (search_ks25.md section 8), one job at a time.
set -e
cd /c/code/github/Kilosort/tuning
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
K='C:\data\ks4_tuning\26648_kobe_2020-08-21'
M='C:\data\ks4_tuning\25843_roger_2019-09-11'
KR="$K\runs"; MR="$M\runs"
LOW_K="$KR\ks25_standard_low_r1 $KR\ks25_standard_low_r2 $KR\ks25_standard_low_r3"
LOW_M="$MR\ks25_standard_low_r1 $MR\ks25_standard_low_r2 $MR\ks25_standard_low_r3"
H8_K="$KR\search\198_H8_cds10_s1\sort $KR\search\208_H8_cds10_s2\sort $KR\search\218_H8_cds10_s3\sort"

# 1. Kobe: each replicate against the five default replicates, logged
for k in 1 2 3; do
  $PY run_candidate.py run --existing-sort "$KR\ks25_standard_low_r$k" --name ks25_standard_low --seed $k --phase comparison \
    --settings '{"sorter": "ks2.5 standard_low"}' \
    --rationale "KS2.5 comparison, preset standard_low (spec v0.20, search_ks25.md section 8): replicate $k against default"
done
# 2. Kobe: H8 against standard_low
k=1
for h in $H8_K; do
  $PY compare.py --candidate "$h" --default $LOW_K --raw-artefacts "$K\search\raw_artefacts_car.npz" --out "$KR\ks25_comparison\H8_vs_ks25low_s$k.json"
  k=$((k + 1))
done
# 3. Kobe: reproducibility and stricter cleaning
$PY reproducibility.py --pair "$KR\ks25_standard_low_r1" "$KR\ks25_standard_low_r2" --pair "$KR\ks25_standard_low_r1" "$KR\ks25_standard_low_r3" \
  --pair "$KR\ks25_standard_low_r2" "$KR\ks25_standard_low_r3" --raw-artefacts "$K\search\raw_artefacts_car.npz" --duration 1200 \
  --hybrid-truth "$K\search\hybrid\hybrid_truth.npz" --out "$KR\reproducibility_ks25low.json"
$PY strict_cleaning.py --sorts $LOW_K --out "$KR\strict_cleaning_ks25low.json"

# 4-5. MEC: standard_low against default, then H8 against standard_low
MEC_ARGS=(--tracking "$M\search\tracking_13575-14775s.npz" --raw-artefacts "$M\search\raw_artefacts_car.npz" --t0 13575 --duration 1200)
$PY transfer.py --default "$MR\default_s1" "$MR\default_s2" "$MR\default_s3" --winner $LOW_M "${MEC_ARGS[@]}" --out "$MR\transfer_ks25low.json"
ALPHA=$($PY -c "import json; print(json.load(open(r'$MR\transfer_ks25low.json'))['alpha_s'])")
$PY transfer.py --default $LOW_M --winner "$MR\H8_s1" "$MR\H8_s2" "$MR\H8_s3" --alpha "$ALPHA" "${MEC_ARGS[@]}" --out "$MR\transfer_H8_vs_ks25low.json"
# 6. MEC: reproducibility
$PY reproducibility.py --pair "$MR\ks25_standard_low_r1" "$MR\ks25_standard_low_r2" --pair "$MR\ks25_standard_low_r1" "$MR\ks25_standard_low_r3" \
  --pair "$MR\ks25_standard_low_r2" "$MR\ks25_standard_low_r3" --raw-artefacts "$M\search\raw_artefacts_car.npz" --duration 1200 \
  --out "$MR\reproducibility_ks25low.json"

# 7. Decoding, all four sorters on the same windows
DEF_K="$KR\hybrid_default_seed1 $KR\hybrid_default_seed2 $KR\hybrid_default_seed3 $KR\hybrid_default_seed4 $KR\hybrid_default_seed5"
STD_K="$KR\ks25_standard_r1 $KR\ks25_standard_r2 $KR\ks25_standard_r3"
DEF_M="$MR\default_s1 $MR\default_s2 $MR\default_s3"
STD_M="$MR\ks25_standard_r1 $MR\ks25_standard_r2 $MR\ks25_standard_r3"
H8_M="$MR\H8_s1 $MR\H8_s2 $MR\H8_s3"
for dec in bayes_decode corr_decode; do
  $PY $dec.py --group default $DEF_K --group ks25 $STD_K --group ks25_low $LOW_K --group H8 $H8_K \
    --tracking "$K\search\tracking_1668-2868s.npz" --raw-artefacts "$K\search\raw_artefacts_car.npz" --t0 1668 --duration 1200 \
    --hybrid-truth "$K\search\hybrid\hybrid_truth.npz" --out "$KR/${dec}_v020.json"
  $PY $dec.py --group default $DEF_M --group ks25 $STD_M --group ks25_low $LOW_M --group H8 $H8_M \
    "${MEC_ARGS[@]}" --out "$MR/${dec}_v020.json"
done
echo all scored
