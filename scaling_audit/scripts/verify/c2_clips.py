import os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, r'C:\code\github\Kilosort')
import numpy as np
import torch
torch.set_num_threads(8)
from kilosort import io
from kilosort.spikedetect import extract_snippets

ops = np.load(r'C:\data\ks4_tuning\26648_kobe_2020-08-21\runs\hybrid_default_seed1\ops.npy', allow_pickle=True).item()
dev = torch.device('cpu')
pp = ops['preprocessing']
hp = torch.as_tensor(np.asarray(pp['hp_filter'])).to(dev)
wm = torch.as_tensor(np.asarray(pp['whiten_mat'])).to(dev)
bfile = io.BinaryFiltered(ops['filename'], ops['n_chan_bin'], ops['fs'], ops['batch_size'], ops['nt'],
                          ops['nt0min'], ops['chanMap'], hp_filter=hp, whiten_mat=wm, device=dev,
                          do_CAR=ops['do_CAR'], invert_sign=ops['invert_sign'])
print('n_batches', bfile.n_batches)
counts = []
t0 = time.time()
for j in range(0, bfile.n_batches, 25):
    X = bfile.padded_batch_to_torch(j, ops)
    c = extract_snippets(X, nt=ops['nt'], twav_min=ops['nt0min'], Th_single_ch=ops['settings']['Th_single_ch'], device=dev)
    counts.append(len(c))
counts = np.array(counts)
print('sampled batches', len(counts), 'clips per sampled batch: mean %.0f min %d max %d' % (counts.mean(), counts.min(), counts.max()))
print('cumulative clips over the 20 min (every 25th batch): %d (buffer 500000)' % counts.sum())
nfill = 500000 / counts.mean()
print('buffer fills after ~%.0f sampled batches = %.0f batches = %.1f min of recording' % (nfill, nfill*25, nfill*25*2/60))
print('elapsed %.0fs' % (time.time()-t0))
