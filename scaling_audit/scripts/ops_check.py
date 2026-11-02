import numpy as np, sys
ops = np.load(sys.argv[1], allow_pickle=True).item()
print(sorted(ops.keys()))
d = ops.get('dshift'); print('dshift', None if d is None else (d.shape, d.min(), d.max(), np.abs(np.diff(d[:,0])).max()))
yc = ops['yc']; print('yc', yc.min(), yc.max(), len(yc), 'dmin', ops.get('dmin'), 'Nbatches', ops.get('Nbatches'))
print('yblk', ops.get('yblk'))
for k in ['Wrot','wTEMP','wPCA']:
    v = ops.get(k); print(k, type(v), getattr(v,'shape',None))
if d is not None:
    x = d[:,0]; print('dshift quartiles per 5 min:', [round(float(np.median(x[i:i+150])),1) for i in range(0,len(x),150)])
