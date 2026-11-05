"""How many spikes does a kNN graph need? Two units, separation D in noise SDs, d feature dims, n spikes each, k=10
neighbours, no downsampling. Prints the fraction of a spike's neighbours from its own unit (0.5 = no information).
This bounds any scheme that re-clusters inside a small batch or window."""
import numpy as np
import torch

torch.set_num_threads(4)
r = np.random.default_rng(0)
k = 10
print('d   D    ' + ''.join(f'n={n:<7d}' for n in [10, 30, 100, 300, 1000, 3000, 10000, 30000]))
for d in [60, 20]:
    for D in [3.2, 4.5, 5.5, 7, 10]:
        row = []
        for n in [10, 30, 100, 300, 1000, 3000, 10000, 30000]:
            vals = []
            for rep in range(max(1, 200 // n + 1) if n < 300 else 1):
                A = r.standard_normal((n, d)).astype('float32')
                B = r.standard_normal((n, d)).astype('float32'); B[:, 0] += D
                X = torch.from_numpy(np.vstack((A, B)))
                lab = torch.cat((torch.zeros(n), torch.ones(n)))
                q = torch.from_numpy(r.choice(2 * n, min(2 * n, 3000), replace=False))
                dist = torch.cdist(X[q], X)
                dist[torch.arange(len(q)), q] = 1e9
                nn = dist.topk(min(k, 2 * n - 1), largest=False)[1]
                vals.append(float((lab[nn] == lab[q][:, None]).float().mean()))
            row.append(np.mean(vals))
        print(f'{d:2d} {D:4.1f}  ' + ''.join(f'{x:<9.3f}' for x in row), flush=True)
