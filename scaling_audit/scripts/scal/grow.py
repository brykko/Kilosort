# Reproduce template_matching.extract's tF growth on CPU and measure peak working set.
import psutil, torch, numpy as np, os
torch.set_num_threads(1)
p = psutil.Process()
def gb(x): return x/2**30
base = p.memory_info().wset
cap = 2*10**6; tF = torch.zeros((cap, 10, 6))      # stand-in for 10**6 start, scaled
tF[:] = 1
before = p.memory_info()
# the doubling line, verbatim
tF = torch.cat((tF, torch.zeros_like(tF)), 0)
after = p.memory_info()
print('old cap GB', gb(cap*240), 'new cap GB', gb(tF.numel()*4))
print('peak_wset over base GB', gb(after.peak_wset-base), ' (model: 4*old =', gb(4*cap*240), ')')
k = int(1.1*cap); isort = np.argsort(np.random.rand(k))
tF[:k] = 2
m0 = p.memory_info().wset
tF2 = tF[isort]                                     # extract's final gather while capacity is alive
print('gather adds GB', gb(p.memory_info().wset-m0), '(model k*240 =', gb(k*240), ')')
