"""E1: default KS4 sorts of time windows of the hybrid excerpt (tmin/tmax), one at a time."""
import json
import subprocess
import sys
import time
from pathlib import Path

PY = r'C:\Users\richa\miniconda3\envs\kilosort\python.exe'
REPO = Path(r'C:\code\github\Kilosort')
EX = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21')
OUT = EX / 'runs' / 'scaling_audit'
OUT.mkdir(parents=True, exist_ok=True)

runs = [('full_0-1200_s1', 0, None)]
runs += [(f'w5_{a}-{a+300}_s1', a, a + 300) for a in (0, 300, 600, 900)]
runs += [(f'w10_{a}-{a+600}_s1', a, a + 600) for a in (0, 600)]
only = set(sys.argv[1:])
for name, tmin, tmax in runs:
    if only and name not in only:
        continue
    out = OUT / name
    if out.exists():
        print(f'skip {name} (exists)', flush=True)
        continue
    s = {'tmin': tmin} | ({'tmax': tmax} if tmax is not None else {})
    t0 = time.time()
    with open(OUT / f'{name}.log', 'w') as log:
        r = subprocess.run([PY, str(REPO / 'tuning' / 'run_sort.py'), '--data', str(EX / 'search/hybrid/ap_1668-2868s_hybrid.bin'),
                            '--probe', str(EX / 'search/probe_3a_opt3.json'), '--n-chan-bin', '385',
                            '--out', str(out), '--seed', '1', '--settings', json.dumps(s)],
                           cwd=REPO / 'tuning', stdout=log, stderr=subprocess.STDOUT)
    print(f'{name}: exit {r.returncode}, {(time.time() - t0) / 60:.1f} min', flush=True)
