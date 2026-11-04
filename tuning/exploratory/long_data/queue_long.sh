#!/bin/bash
# Kobe's 2-hour segment as downloaded (raw, no hybrid units), one sort at a time, seed 1: KS4 default and H8
# with per-spike scores (../autocurate/ks4_sort_scored.py, settings and probe copied from the 20-min sorts),
# then KS2.5 standard and standard_low (ks25_sort.py). KS2.5 writes a whitened copy as large as the segment,
# so it waits for the MEC download to finish. Memory is logged throughout (memlog.py).
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
K=C:/data/ks4_tuning/26648_kobe_2020-08-21
SEG=$K/search/long/ap_354-7554s.bin
MEC_DONE=C:/data/ks4_tuning/25843_roger_2019-09-11/search/long/ap_7575-14775s.manifest.json
O=C:/data/ks4_tuning/exploratory/long_data
mkdir -p "$O/sorts"
cd "$(dirname "$0")"
rm -f "$O/memlog.stop"
$PY memlog.py --out "$O/memlog.log" --stop "$O/memlog.stop" &

for pair in "$K/runs/hybrid_default_seed1 kobe_default_s1" "$K/runs/search/198_H8_cds10_s1/sort kobe_H8_s1"; do
    set -- $pair
    echo "=== $2 (like $1) $(date)"
    $PY ../autocurate/ks4_sort_scored.py --like "$1" --data "$SEG" --out "$O/sorts/$2" --no-clean > "$O/sorts/$2.log" 2>&1 \
        || echo "FAILED: $2"
    tail -1 "$O/sorts/$2.log"
done
echo "=== waiting for the MEC download $(date)"
while [ ! -f "$MEC_DONE" ]; do sleep 60; done
for preset in standard standard_low; do
    echo "=== kobe_ks25_${preset}_s1 $(date)"
    $PY ../../ks25_sort.py --data "$SEG" --meta "$K/search/source.ap.meta" --probe "$K/search/probe_3a_opt3.json" \
        --out "$O/sorts/kobe_ks25_${preset}_s1" --seed 1 --preset $preset > "$O/sorts/kobe_ks25_${preset}_s1.log" 2>&1 \
        || echo "FAILED: kobe_ks25_${preset}_s1"
    tail -1 "$O/sorts/kobe_ks25_${preset}_s1.log"
done
touch "$O/memlog.stop"
echo "=== done $(date)"
