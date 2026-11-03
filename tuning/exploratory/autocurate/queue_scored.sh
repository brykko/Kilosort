#!/bin/bash
# The four KS4 sorts with per-spike scores (ks4_sort_scored.py), one at a time: default and H8, seed 1,
# both excerpts, each copying an existing sort's settings, data and probe.
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
K=C:/data/ks4_tuning/26648_kobe_2020-08-21/runs
M=C:/data/ks4_tuning/25843_roger_2019-09-11/runs
O=C:/data/ks4_tuning/exploratory/autocurate/sorts
mkdir -p "$O"
cd "$(dirname "$0")"
for pair in "$K/hybrid_default_seed1 kobe_default_s1" "$K/search/198_H8_cds10_s1/sort kobe_H8_s1" \
            "$M/default_s1 mec_default_s1" "$M/H8_s1 mec_H8_s1"; do
    set -- $pair
    echo "=== $2 (like $1) $(date)"
    $PY ks4_sort_scored.py --like "$1" --out "$O/$2" > "$O/$2.log" 2>&1 || echo "FAILED: $2"
    tail -2 "$O/$2.log"
done
echo "=== done $(date)"
