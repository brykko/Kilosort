import re, numpy as np, sys
log = sys.argv[1]
p=0; rows=[]
for line in open(log):
    if 'First clustering' in line: p=1
    if 'Final clustering' in line: p=2
    m=re.search(r'Xd shape: torch.Size\(\[(\d+), (\d+)\]\)',line)
    if m: rows.append((p,int(m.group(1)),int(m.group(2))))
for nskip in (20,10):
  base=None
  for f in (1,3,9,15):
    tot=0; capped=0; maxn=0
    for ph in (1,2):
        r=np.array([x[1:] for x in rows if x[0]==ph]); n=r[:,0]*f; d=r[:,1]
        tot += (n*np.minimum(n/nskip,25000)*d).sum()
        capped += (n/nskip>25000).sum(); maxn=max(maxn,n.max())
    base = base or tot
    print(f'nskip={nskip} f={f}: faiss flops-ish ratio vs f=1 {tot/base:6.1f}  centers capped {capped}/192  max center {maxn/1e6:.2f}M')
