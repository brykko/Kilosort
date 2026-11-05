"""Pick R's extension by the preregistered objective on seeds 1-2; ties or no gain keep the tuned R."""
import glob, json, os
import numpy as np
OUT = 'C:/data/ks4_tuning/exploratory/interleaving/simulation'
sc = {}
for f in glob.glob(os.path.join(OUT, 'ext1', '*.json')):
    d = json.load(open(f))
    a = d['arms'][0]
    r = [x for x in a['records'] if x['tag'] == 'round3'][-1]['C']
    sc.setdefault(a['name'], []).append(r['acc'] - .5 * r['n_false'] / 48)
m = {k: float(np.mean(v)) for k, v in sc.items()}
print(m)
best = max(m, key=m.get)
ov = {'Rx_b3k8': dict(bim=.3, K_init=8), 'Rx_b4k4': dict(bim=.4, K_init=4), 'Rx_b3k4': dict(bim=.3, K_init=4)}[best]
print(best, ov)
json.dump(ov, open(os.path.join(OUT, 'ext_R.json'), 'w'))
