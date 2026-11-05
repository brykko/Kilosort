"""Clustering cost against the number of spikes, measured under load. clustering_qr.run on subsets of the round-2
spikes of kobe H8 seed 1 (stage cache, not re-extracted): every k-th batch (D3) or a contiguous buffer (D1, D2), with
timers around its parts and cluster_downsampling varied. Patches are in this process only.

cluster_bench.py ran with default threads and stalled on this busy machine (OpenMP threads spinning at barriers while
other jobs hold the cores), so here threads are limited and passive, and a calibration set (ten centres whose idle
time is in the uncached sort's log) is repeated between runs: scale = measured / idle gives the slowdown at that
moment, and wall_idle_est is wall / scale. Results are saved after each run. Output: cluster_bench2.json."""
import os
os.environ.update(OMP_NUM_THREADS='8', OMP_WAIT_POLICY='PASSIVE', KMP_BLOCKTIME='0')
from common import *
from kilosort import clustering_qr as cq, hierarchical, swarmsplitter
import faiss, psutil, re
from datetime import datetime
faiss.omp_set_num_threads(8)

ops, bfile, folder = load_sort()
d = next(CACHE.glob('B_f42d66d3d2b13925*'))
c = torch.load(d / 'extract.pt', weights_only=False)
ops.update(c['ops'])
st, tF = c['out'][0], c['out'][1]
del c
NT = ops['batch_size']
ib = (st[:, 0] // NT).astype(np.int64)
tpl = st[:, 1].astype(np.int64)
print(st.shape, flush=True)

# calibration set and its idle time, from the sort without stage cache (round 2 pass)
CAL = list(range(40, 50))
pat = re.compile(r'^(\S+ \S+) kilosort.clustering_qr DEBUG\s+Center (\d+) \|')
rows = []
for line in open('C:/data/ks4_tuning/exploratory/iteration/memtest/nosave/kilosort4.log', encoding='utf8', errors='ignore'):
    m = pat.match(line)
    if m:
        rows.append((datetime.strptime(m[1], '%Y-%m-%d %H:%M:%S,%f'), int(m[2])))
rows = rows[96:192]        # second pass of the file
assert rows[0][1] == rows[0][1]
idle = sum((rows[i + 1][0] - rows[i][0]).total_seconds() for i in range(len(rows) - 1) if rows[i][1] in CAL)
xy, iC = cq.xy_templates(ops)
near, _, _ = cq.get_nearest_centers(xy, cq.x_centers(ops), cq.y_centers(ops))
cal_t = np.isin(near.numpy(), CAL)
cal_sel = cal_t[tpl]
print('calibration idle s', idle, 'spikes', int(cal_sel.sum()), flush=True)
T = {}


def timed(mod, name):
    f = getattr(mod, name)
    def g(*a, **k):
        t = time.perf_counter()
        r = f(*a, **k)
        torch.cuda.synchronize()
        T[name] = T.get(name, 0) + time.perf_counter() - t
        return r
    setattr(mod, name, g)


for mod, name in ((cq, 'get_data_cpu'), (cq, 'neigh_mat'), (cq, 'kmeans_plusplus'), (cq, 'cluster'), (hierarchical, 'maketree'), (swarmsplitter, 'split')):
    timed(mod, name)


def run(label, sel, cds=10):
    T.clear()
    ops['settings']['cluster_downsampling'] = cds
    s, f = st[sel], tF[torch.from_numpy(sel)]
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1); np.random.seed(1)
    psutil.cpu_percent(None); c0 = psutil.Process().cpu_times()
    t0 = now()
    clu, Wall = cq.run(ops, s, f, mode='template', device=DEVICE)
    t = now() - t0
    c1 = psutil.Process().cpu_times()
    return {'label': label, 'cds': cds, 'spikes': int(sel.sum()), 'wall_s': t, 'us_per_spike': t / len(s) * 1e6, 'clusters': int(clu.max() + 1),
            'parts_s': {k: float(v) for k, v in T.items()}, 'gpu_peak_GB': torch.cuda.max_memory_allocated() / 2**30,
            'process_cpu_s': c1.user + c1.system - c0.user - c0.system, 'system_cpu_pct': psutil.cpu_percent(None)}


res = {'calibration_idle_s': idle, 'calibration': [], 'runs': []}


def cal():
    r = run('calibration', cal_sel)
    res['calibration'].append(r)
    print('CAL', round(r['wall_s'], 1), 'scale', round(r['wall_s'] / idle, 2), flush=True)
    return r['wall_s'] / idle


def go(label, sel, cds=10):
    a = cal()
    r = run(label, sel, cds)
    r['scale_before'] = a
    r['wall_idle_est_s'] = r['wall_s'] / a
    res['runs'].append(r)
    print(json.dumps(r), flush=True)
    save_json('cluster_bench2.json', res)


go('rolling_30', (ib >= 300) & (ib < 330))
go('every_30', ib % 30 == 0)
go('rolling_60', (ib >= 300) & (ib < 360))
go('every_16', ib % 16 == 0)
go('every_8', ib % 8 == 0)
go('rolling_120', (ib >= 300) & (ib < 420))
go('every_4', ib % 4 == 0)
go('every_4', ib % 4 == 0, cds=20)
go('every_4', ib % 4 == 0, cds=2)
go('every_30', ib % 30 == 0, cds=1)
go('every_2', ib % 2 == 0)
cal()
save_json('cluster_bench2.json', res)
