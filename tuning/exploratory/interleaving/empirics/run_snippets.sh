#!/bin/bash
# The CPU snippet passes, two lanes side by side (each about 7-15 min).
PY=C:/Users/richa/miniconda3/envs/kilosort/python.exe
O=C:/data/ks4_tuning/exploratory/interleaving/empirics
( $PY snippets.py --family kobe_H8 > $O/snippets_kobe_H8.log 2>&1; $PY snippets.py --family kobe_default > $O/snippets_kobe_default.log 2>&1; $PY snippets.py --family kobe_2h > $O/snippets_kobe_2h.log 2>&1 ) &
( $PY snippets.py --family mec_H8 > $O/snippets_mec_H8.log 2>&1; $PY snippets.py --family mec_default > $O/snippets_mec_default.log 2>&1; $PY snippets.py --family mec_2h > $O/snippets_mec_2h.log 2>&1 ) &
wait
