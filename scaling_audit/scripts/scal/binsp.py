# Per-iteration cost of datashift.bin_spikes' loop body ((batch_id==t).nonzero() over all spikes) vs N
import numpy as np, time, os
os.environ['OMP_NUM_THREADS']='1'
rng=np.random.default_rng(0)
for f,label in ((1,'20m'),(9,'3h'),(15,'5h')):
    nb=600*f; n=int(4.02e6*f)
    bid=np.sort(rng.integers(0,nb,n)).astype(np.float64)
    t=time.perf_counter(); K=20
    for j in range(K): ix=(bid==j*(nb//K)).nonzero()[0]
    dt=(time.perf_counter()-t)/K
    print(f'{label}: N={n/1e6:.1f}M nb={nb}: {dt*1e3:.1f} ms/iter -> loop ~{dt*nb:.0f} s')
    del bid
