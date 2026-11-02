"""CPU Monte Carlo: swarmsplitter.check_split bimodality score vs number of spikes in the node.
(a) one Gaussian cloud split in two by a hyperplane (what a graph partition of one unit looks like);
(b) two Gaussians at separation d (sigma units) with true labels, size ratio 1 and 0.1.
D = feature dimension (nchan x 6 PCs)."""
import numpy as np, sys
sys.path.insert(0, r'C:\code\github\Kilosort')
from kilosort.swarmsplitter import bimod_score

rng = np.random.default_rng(1)
D = 90

def score(X, lab):
    # same maths as swarmsplitter.check_split
    Xs = X.copy(); Xs[:, -1] = 1
    labels = 2 * lab - 1
    w = np.ones((Xs.shape[0], 1)); w[labels > 0] = np.mean(labels < 0); w[labels < 0] = np.mean(labels > 0)
    CC = Xs.T @ (Xs * w) + .01 * np.eye(Xs.shape[1])
    b = np.linalg.solve(CC, labels @ (Xs * w))
    return bimod_score(Xs @ b)

def stat(f, reps=30):
    s = np.array([f() for _ in range(reps)]); return f'{np.median(s):.2f} (P>=0.6: {np.mean(s >= 0.6):.2f})'

for n in (150, 300, 600, 1200, 3000, 10000, 30000):
    def uni():
        X = rng.standard_normal((n, D)) * 1.0
        return score(X, (X[:, 0] > 0).astype(int))
    def uni_offcentre():   # unequal split of one unit (30/70)
        X = rng.standard_normal((n, D))
        return score(X, (X[:, 0] > 0.52).astype(int))
    out = [f'n={n:6d}', 'one unit 50/50:', stat(uni), ' 30/70:', stat(uni_offcentre)]
    for d, ratio in ((2.0, 1.0), (3.0, 1.0), (3.0, 0.1), (4.0, 0.1)):
        def two():
            n2 = max(int(n * ratio / (1 + ratio)), 5); n1 = n - n2
            X = rng.standard_normal((n, D)); X[n1:, 0] += d
            lab = np.r_[np.zeros(n1), np.ones(n2)].astype(int)
            return score(X, lab)
        out += [f' d={d},r={ratio}:', stat(two, 15)]
    print(' '.join(out), flush=True)
