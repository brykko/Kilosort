"""Rebuild per-spike PC features (10 nearest chans of the unit's peak channel x 6 PCs) from the
preprocessed, drift-corrected data on CPU, for every `step`-th batch. Features of the data X (not
the matching residual), so overlapping spikes are not removed.
usage: recon_feats.py RUN_DIR OUT.npz STEP [MAXBATCH]"""
import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:/code/github/Kilosort')
import numpy as np, torch
torch.set_num_threads(8)
from kilosort.io import load_ops, bfile_from_ops
run, out, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
maxb = int(sys.argv[4]) if len(sys.argv) > 4 else None
cpu = torch.device('cpu')
ops = load_ops(os.path.join(run, 'ops.npy'), device=cpu)
bfile = bfile_from_ops(ops, device=cpu)
st = np.load(os.path.join(run, 'spike_times.npy')).astype(np.int64) - int(bfile.imin)
clu = np.load(os.path.join(run, 'spike_clusters.npy')).astype(np.int64)
tmpl = np.load(os.path.join(run, 'templates.npy'))          # (n_units, nt, Nchan)
peak = (tmpl**2).sum(1).argmax(1)
iCC = ops['iCC'].cpu().numpy() if torch.is_tensor(ops['iCC']) else ops['iCC']
chans = iCC[:, peak].T                                        # (n_units, 10)
wPCA = ops['wPCA'].float().cpu()
NT, nt, nt0min = int(ops['batch_size']), int(ops['nt']), int(ops['nt0min'])
nb = int(bfile.n_batches); batches = np.arange(0, nb, step)
if maxb: batches = batches[:maxb]
bstart = np.searchsorted(st, batches*NT); bend = np.searchsorted(st, (batches+1)*NT)
F, T, C = [], [], []
t0 = time.time()
with torch.no_grad():
    for j, ib in enumerate(batches):
        X = bfile.padded_batch_to_torch(int(ib), ops)
        idx = np.arange(bstart[j], bend[j])
        s = st[idx] - ib*NT + nt - nt0min
        ok = (s >= 0) & (s + nt <= X.shape[1]); idx, s = idx[ok], s[ok]
        ch = torch.from_numpy(chans[clu[idx]])                   # (n,10)
        tw = torch.from_numpy(s)[:, None] + torch.arange(nt)     # (n,61)
        W = X[ch[:, :, None], tw[:, None, :]]                    # (n,10,61)
        F.append((W @ wPCA.T).numpy().astype(np.float32)); T.append(st[idx]); C.append(clu[idx])
        if j == 2: print('sec/batch', (time.time()-t0)/3, flush=True)
np.savez(out, F=np.concatenate(F), t=np.concatenate(T), c=np.concatenate(C), batches=batches, NT=NT,
         chans=chans, dur=nb*NT/30000.)
print('done', len(np.concatenate(T)), 'spikes from', len(batches), 'batches in', round(time.time()-t0), 's')
