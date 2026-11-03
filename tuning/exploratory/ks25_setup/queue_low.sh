#!/bin/bash
# KS2.5 "standard_low" sorts (search_ks25.md section 8), one at a time: kobe r1-r3, then MEC r1-r3.
set -e
cd /c/code/github/Kilosort/tuning
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
K='C:\data\ks4_tuning\26648_kobe_2020-08-21'
M='C:\data\ks4_tuning\25843_roger_2019-09-11'
for k in 1 2 3; do
  $PY ks25_sort.py --data "$K\search\hybrid\ap_1668-2868s_hybrid.bin" --meta "$K\search\source.ap.meta" --probe "$K\search\probe_3a_opt3.json" --out "$K\runs\ks25_standard_low_r$k" --seed $k --preset standard_low
done
for k in 1 2 3; do
  $PY ks25_sort.py --data "$M\search\ap_13575-14775s.bin" --meta "$M\search\source.ap.meta" --probe "$M\search\probe_3a_opt3.json" --out "$M\runs\ks25_standard_low_r$k" --seed $k --preset standard_low
done
