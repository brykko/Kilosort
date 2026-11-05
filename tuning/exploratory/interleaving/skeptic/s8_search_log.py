"""A natural experiment in the search log: runs that differ only in merge_correlation_threshold_1 have the same first
clusters but dictionaries of very different size, with no extra round. Reads each run's kilosort4.log for the numbers
of first clusters, extraction templates (the 'iU shape' debug line) and final clusters; S from search_log.jsonl.
Compared only within one spec version (between two freeze records), since S changed between versions."""
import collections
import re

from common import *  # noqa

R = Path('C:/data/ks4_tuning/26648_kobe_2020-08-21/runs')
L = [json.loads(x) for x in (R / 'search_log.jsonl').read_text().splitlines() if x.strip()]
start, res, ver = {}, {}, {}
v = 0
for r in L:
    if r['type'] == 'freeze':
        v += 1
    elif r['type'] == 'start':
        start[r['run_id']] = r
        ver[r['run_id']] = v
    else:
        res[r['run_id']] = r
rows = []
for rid, s in start.items():
    f = R / 'search' / rid / 'sort' / 'kilosort4.log'
    if rid not in res or not f.exists():
        continue
    txt = f.read_text(errors='ignore')
    iu = re.findall(r'iU shape: torch.Size\(\[(\d+)\]\)', txt)
    wall = re.findall(r'Wall shape: torch.Size\(\[(\d+), ', txt)
    if not iu or len(wall) < 2:
        continue
    q = res[rid]
    rows.append(dict(id=rid, seed=s['seed'], ver=ver[rid], settings=s['settings'], S=q.get('S'), dS=q.get('dS'), St=q.get('S_track'),
                     eligible=q.get('n_eligible'), first=int(wall[0]), templates=int(iu[0]), final_clusters=int(wall[1]),
                     units=int(wall[-1])))
print('runs with logs', len(rows))
(OUT / 's8_runs.json').write_text(json.dumps(rows, indent=0, default=str))


def key(r):
    return json.dumps({k: v_ for k, v_ in sorted(r['settings'].items())
                       if k not in ('merge_correlation_threshold_1', 'random_seed', 'cluster_init_seed')}, sort_keys=True, default=str)


g = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    g[(r['ver'], key(r))][r['settings'].get('merge_correlation_threshold_1', 0.9)].append(r)
for k, d in g.items():
    if len(d) > 1:
        print('--', 'spec', k[0], k[1][:170])
        for th, rs in sorted(d.items()):
            sts = [r['St'] for r in rs if r['St'] is not None]
            print(f'  first merge {th}: seeds {[r["seed"] for r in rs]}, S {np.mean([r["S"] for r in rs]):.1f}, '
                  f'S_track {np.mean(sts) if sts else float("nan"):.1f}, first {np.mean([r["first"] for r in rs]):.0f}, '
                  f'templates {np.mean([r["templates"] for r in rs]):.0f}, final clusters {np.mean([r["final_clusters"] for r in rs]):.0f}, '
                  f'units {np.mean([r["units"] for r in rs]):.0f}, eligible {[r["eligible"] for r in rs]}, templates/cluster '
                  f'{np.mean([r["templates"] / r["final_clusters"] for r in rs]):.2f}')
# Last spec version: every run, dictionary against score
last = [r for r in rows if r['ver'] == max(x['ver'] for x in rows)]
print('runs under the last spec version with logs:', len(last))
s = collections.defaultdict(list)
for r in rows:
    s[(r['ver'], key(r), r['settings'].get('merge_correlation_threshold_1', 0.9))].append(r)
print('largest S under the latest versions, with dictionary:')
ag = [(np.mean([r['dS'] for r in v_ if r['dS'] is not None] or [np.nan]), k[0], np.mean([r['templates'] for r in v_]), np.mean([r['final_clusters'] for r in v_]), v_[0]['id']) for k, v_ in s.items() if len(v_) >= 3]
for a in sorted(ag, reverse=True, key=lambda x: (x[1], x[0]))[:14]:
    print(f'  dS {a[0]:7.1f} spec {a[1]} templates {a[2]:6.0f} final clusters {a[3]:6.0f} ratio {a[2] / a[3]:.2f} {a[4]}')
