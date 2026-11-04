#!/bin/bash
# After queue_ks25.sh: all scores of the long-data round (score_long.py), then the tables and figures
# (summarise_long.py).
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
O=C:/data/ks4_tuning/exploratory/long_data
cd "$(dirname "$0")"
echo "=== waiting for queue_ks25.sh $(date)"
until grep -q "=== done" "$O/queue_ks25.log" 2>/dev/null; do sleep 60; done
echo "=== score_long $(date)"
$PY score_long.py
echo "=== summarise_long $(date)"
$PY summarise_long.py
echo "=== done $(date)"
