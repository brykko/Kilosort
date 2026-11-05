"""Does a rolling feature buffer or repeated clustering grow memory under torch's CPU allocator on Windows
(intermediate_results_audit.md)? CPU only. Output: alloc_test.json."""
import json, sys, time
import numpy as np, psutil, torch

p = psutil.Process()
gb = lambda: (p.memory_info().private / 2**30, p.memory_info().rss / 2**30)
res = {'torch': torch.__version__, 'start': gb()}
rng = np.random.default_rng(0)


def loop(name, sizes_mb, numpy=False):
    t = time.time(); peak = 0
    for s in sizes_mb:
        n = int(s * 2**20 / 4)
        if numpy:
            x = torch.from_numpy(np.zeros(n, 'float32')); x[::1024] = 1
        else:
            x = torch.zeros(n); x[::1024] = 1
        peak = max(peak, p.memory_info().private)
        del x
    res[name] = {'n': len(sizes_mb), 'mb': [float(np.min(sizes_mb)), float(np.max(sizes_mb))], 'peak_private_GB': peak / 2**30, 'end_private_GB': gb()[0], 'end_rss_GB': gb()[1], 's': time.time() - t}
    print(name, res[name], flush=True)


# per-centre arrays at the scale of a rolling buffer, then of a whole round, repeated
loop('centre_small_fixed', [8.0] * 3000)
loop('centre_small_varied', list(rng.uniform(1, 30, 3000)))
loop('centre_round_varied', list(rng.uniform(10, 120, 2000)))
loop('buffer_fixed_600MB', [600.0] * 100)
loop('buffer_varied_300_900MB', list(rng.uniform(300, 900, 100)))
loop('buffer_growing_numpy', list(np.linspace(300, 2000, 40)), numpy=True)
loop('buffer_growing_torch', list(np.linspace(300, 2000, 40)))
(open('C:/data/ks4_tuning/exploratory/interleaving/compute/alloc_test.json', 'w')).write(json.dumps(res, indent=1))
