"""Reading and preprocessing a batch in the three orders the designs need: sequential and strided (prefetched by
_BatchReader) and random (no prefetch), on the 20-min and the 2-hour file, with a 170 ms sleep standing in for the
GPU work so that the prefetch thread gets the same time as in a sort. Also a strided extraction pass on real batches.
Output: io_order.json."""
from common import *
from kilosort.io import BinaryRWFile

res = {}
rng = np.random.default_rng(0)
for name in ('kobe_repeat_s1', 'kobe_2h_repeat_s1'):
    ops, bfile, folder = load_sort(name)
    nb = int(bfile.n_batches)
    orders = {'sequential': list(range(nb // 2, nb // 2 + 40)), 'strided_k6': list(range(30, 30 + 40 * 6, 6)),
              'strided_k15': list(range(30, min(nb, 30 + 40 * 15), 15)), 'random': list(rng.integers(0, nb, 40))}
    r = {}
    for k, o in orders.items():
        tr, tf = [], []
        for ib in o:
            t = now()
            X = BinaryRWFile.padded_batch_to_torch(bfile, int(ib)); torch.cuda.synchronize(); t1 = time.perf_counter()
            X = bfile.filter(X, ops, int(ib)); torch.cuda.synchronize(); t2 = time.perf_counter()
            tr.append(t1 - t); tf.append(t2 - t1)
            time.sleep(0.17)
        r[k] = {'read_ms_median': float(np.median(tr[3:]) * 1e3), 'read_ms_p90': float(np.percentile(tr[3:], 90) * 1e3), 'filter_ms': float(np.median(tf[3:]) * 1e3)}
        print(name, k, r[k], flush=True)
    res[name] = r
    del bfile
save_json('io_order.json', res)
