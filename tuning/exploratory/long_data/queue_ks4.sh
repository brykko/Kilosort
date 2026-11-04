#!/bin/bash
# KS4 default and H8 with per-spike scores (../autocurate/ks4_sort_scored.py), seed 1, one sort at a time:
# kobe's raw 20-min excerpt (the length comparison's baseline; the earlier 20-min kobe sorts were of the
# hybrid excerpt), then MEC's 2-hour segment. Settings and probe are copied from the 20-min sorts. Memory is
# logged throughout (memlog.py).
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
K=C:/data/ks4_tuning/26648_kobe_2020-08-21
M=C:/data/ks4_tuning/25843_roger_2019-09-11
O=C:/data/ks4_tuning/exploratory/long_data
mkdir -p "$O/sorts"
cd "$(dirname "$0")"
rm -f "$O/memlog.stop"
$PY memlog.py --out "$O/memlog_ks4.log" --stop "$O/memlog.stop" &

run() {  # like, data, name, extra args
    echo "=== $3 (like $1) $(date)"
    $PY ../autocurate/ks4_sort_scored.py --like "$1" --data "$2" --out "$O/sorts/$3" ${4:-} > "$O/sorts/$3.log" 2>&1 \
        || echo "FAILED: $3"
    tail -1 "$O/sorts/$3.log"
}
run "$K/runs/hybrid_default_seed1" "$K/search/ap_1668-2868s.bin" kobe_default_20min_s1
run "$K/runs/search/198_H8_cds10_s1/sort" "$K/search/ap_1668-2868s.bin" kobe_H8_20min_s1
run "$M/runs/default_s1" "$M/search/long/ap_7575-14775s.bin" mec_default_s1 --no-clean
run "$M/runs/H8_s1" "$M/search/long/ap_7575-14775s.bin" mec_H8_s1 --no-clean
touch "$O/memlog.stop"
echo "=== done $(date)"
