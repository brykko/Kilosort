import os, sys; os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np, torch
for p in sys.argv[1:]:
    c = torch.load(p, weights_only=False, map_location='cpu'); st = c['out'][0]
    th = st[:,2]
    print(p[-40:], 'n', len(th), 'min', round(float(th.min()),3), 'templates', int(st[:,1].max())+1,
          'frac<Th/0.9 %.3f'%(th < th.min()/0.9).mean())
    del c, st
