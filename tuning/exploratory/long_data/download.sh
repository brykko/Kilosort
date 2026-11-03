#!/bin/bash
# The 2-hour segments for the long-data round, one after the other, with tuning/extract_segment.py (byte ranges
# by rclone cat, joined, hashed, spot-checked, with a manifest). Kobe: 354-7554 s (open field 1, the untracked
# gap, then linear track; contains the 1668-2868 s excerpt). MEC: 7575-14775 s (the last 2 h of open-field
# session 1; ends with the 13575-14775 s excerpt).
set -e
cd /c/code/github/Kilosort/tuning
PY=/c/Users/richa/miniconda3/envs/kilosort/python.exe
echo "=== kobe $(date)"
$PY extract_segment.py --remote-probe moser:ro/moser-intdirection/data/neuropixels/26648_kobe/recordings/2020-08-21_20-05-19/probe_1 \
  --t0 354 --t1 7554 --out-dir C:/data/ks4_tuning/26648_kobe_2020-08-21/search/long
echo "=== mec $(date)"
$PY extract_segment.py --remote-probe moser:sweeps/richarga/data/roger_25843/rec/2019-09-11_20-59-06/probe_1 \
  --t0 7575 --t1 14775 --out-dir C:/data/ks4_tuning/25843_roger_2019-09-11/search/long
echo "=== done $(date)"
