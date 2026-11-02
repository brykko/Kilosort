"""First-pass clustering (clustering_qr.run, mode='spikes') on a snapshot: KS4 vs the 'clu' patches."""
import sys
import time
from pathlib import Path

sys.path.insert(0, r'C:\code\github\Kilosort')
import numpy as np  # noqa: E402
import torch  # noqa: E402
from kilosort import io, clustering_qr  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ks_fast  # noqa: E402

run = Path(sys.argv[1])
dev = torch.device('cuda')
ops = io.load_ops(run / 'ops.npy', device=dev)
MODE = sys.argv[2] if len(sys.argv) > 2 else 'spikes'
tag = 'first' if MODE == 'spikes' else 'final'
st = np.load(run / 'snapshot' / f'st_{tag}.npy')
tF = torch.load(run / 'snapshot' / f'tF_{tag}.pt')

orig = (clustering_qr.kmeans_plusplus, clustering_qr.assign_iclust, clustering_qr.assign_isub)
res = {}
for name in ['ks4', 'ks4_again', 'patched']:
    if name == 'patched':
        ks_fast.apply(['clu'])
    torch.cuda.synchronize(); t = time.perf_counter()
    clu, Wall = clustering_qr.run(ops, st, tF, mode=MODE, device=dev)
    torch.cuda.synchronize()
    res[name] = (clu, Wall, time.perf_counter() - t)
    print(f'{name:8s} {res[name][2]:6.1f} s  {clu.max() + 1} clusters')
for other in ['ks4_again', 'patched']:
    (c0, W0, _), (c1, W1, _) = res['ks4'], res[other]
    print(f'ks4 vs {other}: clu identical', np.array_equal(c0, c1),
          ' Wall identical', W0.shape == W1.shape and torch.equal(W0, W1))
