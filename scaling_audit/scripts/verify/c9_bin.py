import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
# cost of one iteration of bin_spikes' per-batch scan at 3 h scale
nsp = int(4.08e6 * 9)  # drift-pass spikes at 3 h (4.08M / 20 min)
nb = 5400
batch_id = np.sort(np.random.randint(0, nb, nsp)).astype('float64')
t0 = time.time()
for t in range(5):
    ix = (batch_id == t).nonzero()[0]
dt = (time.time() - t0) / 5
print(f'{nsp} spikes: {dt*1e3:.1f} ms per batch scan -> {dt*nb:.0f} s total for {nb} batches')
nsp2 = int(4.08e6); nb2 = 600
b2 = batch_id[:nsp2]
t0 = time.time()
for t in range(20):
    ix = (b2 == t).nonzero()[0]
dt2 = (time.time() - t0) / 20
print(f'{nsp2} spikes: {dt2*1e3:.1f} ms per scan -> {dt2*nb2:.1f} s total for {nb2} batches')
