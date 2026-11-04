import numpy as np
from msr import msr, inhom, refractory, multiscale_rate, poisson
print("duration dependence, multiscale unit 5 Hz (5 seeds)")
for dur in [300.0, 1200.0, 3600.0]:
    v=[];pm=[]
    for s in range(5):
        r=np.random.default_rng(s)
        t=refractory(inhom(multiscale_rate(dur,5,r),dur,r),0.002)
        v.append(msr(t,dur)); pm.append(msr(poisson(t.size/dur,dur,r),dur))
    print(f"  dur {dur:6.0f}s M~{5*dur:6.0f} MSR {np.mean(v):.4f}+-{np.std(v):.4f}  Poisson {np.mean(pm):.4f}  excess {np.mean(v)-np.mean(pm):+.4f}")
print("base-grid dependence (20 min)")
dur=1200.0
for base in [1.0, 0.1]:
    v=[]
    for s in range(5):
        r=np.random.default_rng(s)
        t=refractory(inhom(multiscale_rate(dur,5,r),dur,r),0.002)
        tt=np.sort(np.concatenate([t, (lambda q: q+r.uniform(1e-4,5e-4,q.size))(t[r.uniform(size=t.size)<0.1])]))
        v.append((msr(t,dur,base_ms=base), msr(tt,dur,base_ms=base)))
    v=np.array(v)
    print(f"  base {base} ms: clean {v[:,0].mean():.4f}  +10% dup {v[:,1].mean():.4f}  d {np.mean(v[:,1]-v[:,0]):+.4f}")
