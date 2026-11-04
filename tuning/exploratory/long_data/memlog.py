"""Logs memory every 30 s while a queue runs: system RAM in use, and the resident memory of the largest
python or MATLAB process (GB). Stops when the file named by --stop exists.

    python memlog.py --out <log> --stop <file>
"""
import argparse
import time
from pathlib import Path

import psutil

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--stop', required=True)
args = p.parse_args()
with open(args.out, 'a') as f:
    while not Path(args.stop).exists():
        rss = [(q.info['memory_info'].rss, q.info['name']) for q in psutil.process_iter(['name', 'memory_info'])
               if q.info['memory_info'] and q.info['name'] and q.info['name'].lower().startswith(('python', 'matlab'))]
        top = max(rss, default=(0, '-'))
        f.write(f"{time.strftime('%H:%M:%S')}  system {psutil.virtual_memory().used / 2**30:5.1f} GB  "
                f"largest {top[1]} {top[0] / 2**30:5.1f} GB\n")
        f.flush()
        time.sleep(30)
