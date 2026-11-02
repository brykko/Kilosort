import torch, glob, numpy as np
for f in sorted(glob.glob(r'C:/data/ks4_tuning/26648_kobe_2020-08-21/search/stage_cache/A_*/drift.pt')):
    c = torch.load(f, weights_only=False, map_location='cpu')
    st = c['out'][1]; o = c['ops']
    print(f.split('/')[-2], type(st), st.shape, sorted(o.keys()), 'dshift range', o['dshift'].min(), o['dshift'].max(), 'spk/batch', st.shape[0]/600)
