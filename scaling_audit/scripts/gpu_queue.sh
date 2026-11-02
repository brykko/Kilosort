#!/bin/bash
# Run the partial-sort GPU tests one at a time.
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
S='C:\Users\richa\AppData\Local\Temp\claude\C--code-github-Kilosort\06f0964c-a905-4a5f-bc87-94db003e3883\scratchpad'
K=C:/data/ks4_tuning/26648_kobe_2020-08-21
M=C:/data/ks4_tuning/25843_roger_2019-09-11
OUT=$K/runs/scaling_audit
run() {  # name mode data probe
  if [ -e "$OUT/$1" ]; then echo "skip $1"; return; fi
  s=$(date +%s)
  "$PY" "$S\\gpu_tests.py" "$2" "$3" "$4" "$OUT/$1" > "$OUT/$1.log" 2>&1
  echo "$1: exit $? in $(( ($(date +%s) - s) / 60 )) min"
}
run p1sweep_kobe sweep $K/search/hybrid/ap_1668-2868s_hybrid.bin $K/search/probe_3a_opt3.json
run tage_kobe_early early $K/search/hybrid/ap_1668-2868s_hybrid.bin $K/search/probe_3a_opt3.json
run tage_kobe_late late $K/search/hybrid/ap_1668-2868s_hybrid.bin $K/search/probe_3a_opt3.json
run tage_mec_early early $M/search/ap_13575-14775s.bin $M/search/probe_3a_opt3.json
run tage_mec_late late $M/search/ap_13575-14775s.bin $M/search/probe_3a_opt3.json
