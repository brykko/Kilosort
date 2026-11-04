#!/bin/bash
# KS2.5 standard and standard_low (ks25_sort.py), seed 1, one sort at a time, after queue_ks4.sh: kobe's raw
# 20-min excerpt (the length comparison's baseline), then kobe's and MEC's 2-hour segments. A long KS2.5
# sort writes a whitened copy as large as the segment (166 GB; deleted at the end). Memory is logged.
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
K=C:/data/ks4_tuning/26648_kobe_2020-08-21
M=C:/data/ks4_tuning/25843_roger_2019-09-11
O=C:/data/ks4_tuning/exploratory/long_data
cd "$(dirname "$0")"
echo "=== waiting for queue_ks4.sh $(date)"
until grep -q "=== done" "$O/queue_ks4.log" 2>/dev/null; do sleep 60; done
rm -f "$O/memlog.stop"
$PY memlog.py --out "$O/memlog_ks25.log" --stop "$O/memlog.stop" &

run() {  # excerpt dir, data, name, preset
    echo "=== $3 $(date)"
    $PY ../../ks25_sort.py --data "$2" --meta "$1/search/source.ap.meta" --probe "$1/search/probe_3a_opt3.json" \
        --out "$O/sorts/$3" --seed 1 --preset $4 > "$O/sorts/$3.log" 2>&1 || echo "FAILED: $3"
    tail -1 "$O/sorts/$3.log"
}
for preset in standard standard_low; do
    run $K "$K/search/ap_1668-2868s.bin" kobe_ks25_${preset}_20min_s1 $preset
done
for preset in standard standard_low; do
    run $K "$K/search/long/ap_354-7554s.bin" kobe_ks25_${preset}_s1 $preset
done
for preset in standard standard_low; do
    run $M "$M/search/long/ap_7575-14775s.bin" mec_ks25_${preset}_s1 $preset
done
touch "$O/memlog.stop"
echo "=== done $(date)"
