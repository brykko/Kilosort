import numpy as np, torch, sys, time
torch.set_num_threads(8)
from kilosort import io, preprocessing, spikedetect
ops = np.load(sys.argv[1], allow_pickle=True).item()
dev = torch.device('cpu')
hp = preprocessing.get_highpass_filter(30000, 300, device=dev)
W = torch.from_numpy(np.asarray(ops['Wrot'])).float()
bf = io.BinaryFiltered(sys.argv[2], 385, 30000, 60000, 61, 20, ops['chanMap'], hp_filter=hp, whiten_mat=W, device=dev)
print('n_batches', bf.n_batches)
for j in [0, 100, 200, 300, 400, 500, 575]:
    X = bf.padded_batch_to_torch(j)
    c = spikedetect.extract_snippets(X, nt=61, twav_min=20, Th_single_ch=6, device=dev)
    print(j, len(c))
