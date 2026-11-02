"""Collect per-stage counts, runtimes and memory from every kilosort4.log under the two datasets' runs folders."""
import json
import re
from pathlib import Path

import pandas as pd

ROOTS = [Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs'), Path('C:/data/ks4_tuning/25843_roger_2019-09-11/runs')]
STEPS = ['preprocessing', 'drift corr', 'spike det. (univ)', 'cluster (temp)', 'spike det. (learn)',
         'cluster (final)', 'cluster merge', 'postprocessing']
rows = []
for root in ROOTS:
    for log in root.rglob('kilosort4.log'):
        txt = log.read_text(errors='ignore')
        if 'Total runtime' not in txt:
            continue
        d = log.parent
        r = {'dataset': root.parent.name[:5], 'run': str(d.relative_to(root))}
        g = lambda pat, f=float: (lambda m: f(m.group(1)) if m else None)(re.search(pat, txt))
        r['n_batches'] = g(r'N batches: (\d+)', int)
        sp = re.findall(r'(\d+) spikes extracted', txt)
        cl = re.findall(r'(\d+) clusters found', txt)
        r['spk_univ'], r['spk_learn'] = (int(sp[0]) if sp else None), (int(sp[1]) if len(sp) > 1 else None)
        r['clu_1'], r['clu_2'] = (int(cl[0]) if cl else None), (int(cl[1]) if len(cl) > 1 else None)
        r['units'] = g(r'(\d+) units found, in', int)
        r['good'] = g(r'(\d+) units found with good', int)
        r['drift_um'] = g(r'Mean abs. drift\s+([\d.]+)')
        r['total_s'] = g(r'Total runtime: ([\d.]+)s')
        for s in STEPS:
            k = re.escape(s)
            r['t_' + s] = g(k + r':\s+([\d.]+)s')
            r['sys_' + s] = g(k + r':\s+sys\s+([\d.]+) GB')
            r['gpu_' + s] = g(k + r':\s+sys\s+[\d.]+ GB\s+\|\s+gpu\s+([\d.]+) GB')
        prov = next((p for p in (d / 'provenance.json', d.parent / 'provenance.json') if p.exists()), None)
        if prov:
            pj = json.loads(prov.read_text())
            st = {k: v for k, v in pj.get('settings', {}).items() if k not in ('n_chan_bin', 'fs')}
            r['settings'] = json.dumps(st, sort_keys=True)
            r['kernels'] = pj.get('kernels_used')
            r['cache'] = bool(pj.get('stage_cache'))
        summ = d.parent / 'summary.json'
        if summ.exists():
            sj = json.loads(summ.read_text())
            r['dS'] = sj.get('dS')
            r['hyb_rec'] = (sj.get('hybrid') or {}).get('n_recovered')
        rows.append(r)
df = pd.DataFrame(rows)
out = Path(r'C:\Users\richa\AppData\Local\Temp\claude\C--code-github-Kilosort\06f0964c-a905-4a5f-bc87-94db003e3883\scratchpad\ks4_logs.csv')
df.to_csv(out, index=False)
pd.set_option('display.width', 250, 'display.max_columns', 60, 'display.max_colwidth', 60)
print(len(df), 'logs;', df.groupby('dataset').size().to_dict())
print(df.groupby('dataset')[['n_batches', 'spk_univ', 'spk_learn', 'clu_1', 'clu_2', 'units', 'good', 'drift_um', 'total_s']].describe().T.round(1))
nc = df[df.cache != True]
print('\nuncached runs:', len(nc))
cols = [c for c in df if c.startswith(('t_', 'sys_', 'gpu_'))]
print(nc.groupby('dataset')[cols].median().T.round(1))
