"""Clustering cost per centre from the timestamps of the 'Center i | Xd shape' DEBUG lines in kilosort4.log, for each
clustering pass of the 20-min and 2-hour sorts, and memory lines. Output: center_times.json."""
import json, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np

S = Path('C:/data/ks4_tuning/exploratory/iteration/sorts')
L = Path('C:/data/ks4_tuning/exploratory/long_data/sorts')
pat = re.compile(r'^(\S+ \S+) kilosort.clustering_qr DEBUG\s+Center (\d+) \| Xd shape: torch.Size\(\[(\d+), (\d+)\]\) \| ntemp: (\d+)')
stage = re.compile(r'^(\S+ \S+) kilosort.run_kilosort INFO\s+(First clustering|Final clustering.*|.*clusters found.*)')
fmt = '%Y-%m-%d %H:%M:%S,%f'
out = {}
for name, f in (('20min_uncached', Path('C:/data/ks4_tuning/exploratory/iteration/memtest/nosave/kilosort4.log')),
                ('2h', S / 'kobe_2h_repeat_s1/kilosort4.log')):
    passes = []; cur = None
    for line in open(f, encoding='utf8', errors='ignore'):
        m = stage.match(line)
        if m:
            t = datetime.strptime(m[1], fmt)
            if 'clustering' in m[2] and 'found' not in m[2]:
                cur = {'label': m[2], 't0': t, 'c': []}
            elif cur is not None:
                cur['t1'] = t; passes.append(cur); cur = None
            continue
        m = pat.match(line)
        if m and cur is not None:
            cur['c'].append((datetime.strptime(m[1], fmt), int(m[3]), int(m[4]), int(m[5])))
    res = []
    for p in passes:
        c = p['c']
        if not c:
            continue
        # a centre's line is logged after get_data_cpu, so dt to the next line = clustering of it + get_data of the next
        t = np.array([(x[0] - p['t0']).total_seconds() for x in c] + [(p['t1'] - p['t0']).total_seconds()])
        dt = np.diff(t); n = np.array([x[1] for x in c], float); d = np.array([x[2] for x in c], float)
        ok = n > 0
        A = np.c_[np.ones(ok.sum()), n[ok]]
        coef, *_ = np.linalg.lstsq(A, dt[ok], rcond=None)
        A2 = np.c_[np.ones(ok.sum()), n[ok], n[ok]**2]
        coef2, *_ = np.linalg.lstsq(A2, dt[ok], rcond=None)
        p2 = np.polyfit(np.log(n[ok]), np.log(dt[ok]), 1)
        res.append({'label': p['label'], 'total_s': float(t[-1]), 'centres': len(c), 'spikes_median': float(np.median(n)), 'spikes_p95': float(np.percentile(n, 95)),
                    'spikes_max': float(n.max()), 'dim_median': float(np.median(d)), 'dt_median': float(np.median(dt)), 'dt_p95': float(np.percentile(dt, 95)),
                    'dt_max': float(dt.max()), 'fit_a_s': float(coef[0]), 'fit_b_us_per_spike': float(coef[1] * 1e6), 'loglog_slope': float(p2[0]),
                    'top10pct_share_time': float(np.sort(dt)[-len(dt)//10:].sum() / dt.sum()),
                    'total_spikes_in_centres': float(n.sum()), 'us_per_spike_overall': float(t[-1] / n.sum() * 1e6), 'n_gt_1M': int((n > 1e6).sum()), 'n_gt_250k': int((n > 2.5e5).sum())})
    out[name] = res
    for r in res:
        print(name, json.dumps(r))
Path('C:/data/ks4_tuning/exploratory/interleaving/compute/center_times.json').write_text(json.dumps(out, indent=1))
