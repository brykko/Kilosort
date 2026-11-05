"""Cost model for the round designs. Stage times are measured (sort logs, no stage cache, no round saves); per-batch
and per-update costs come from this folder's benchmarks; everything put together is an estimate. Prints a table and
writes cost_model.json. Clustering of a subset of N spikes is 3 s + argv[1] microseconds per spike (default 10.5, the
full 20-min rate; cluster_bench2 agrees within about +-50% down to 0.4M spikes on a loaded machine)."""
import json, sys
from pathlib import Path

OUT = Path('C:/data/ks4_tuning/exploratory/interleaving/compute')
US = float(sys.argv[1]) if len(sys.argv) > 1 else 10.5
L = {
    '20min': dict(nb=600, pre=0.97, drift=33.1, univ=32.0, clu1=73.8, ext=[133.6, 108.4, 106.2], clu=[124.2, 122.9, 127.8], merge=8.2, export=7.0,
                  total4=888.2, total2=389.8, n=[2504, 1767, 1716], N=11.9e6, nuniv=5.32e6),
    '2h': dict(nb=3600, pre=3.06, drift=220.4, univ=173.6, clu1=498.4, ext=[872.2, 659.7, 649.8], clu=[924.3, 901.7, 925.9], merge=64.0, export=47.9,
               total4=6005.6, total2=2882.8, n=[3003, 1969, 1924], N=69e6, nuniv=31.2e6),
}
e = lambda n: 0.074 + 5.4e-5 * n          # s per batch of extraction, fit to 1795 and 2504 templates, checked on the logs
ctc = lambda n: 16e-3 * (n / 1767)**1.8   # s per prepare_matching (measured, 1000 to 3933 templates)
UPD = 1.8e-3                              # s per batch: running means of the templates


def online(n, R=10, birth=7e-3, merge=15e-3):
    """s per batch of an online step, dictionary and ctc updated every R batches; birth = residual detection every
    5th batch (estimate from universal detection, 35 ms per batch); merge = one template merge per 100 batches."""
    return e(n) + UPD + ctc(n) / R + birth + merge


def base(l):
    return l['pre'] + l['drift']


def final(l):
    return l['ext'][-1] + l['clu'][-1] + l['merge'] + l['export']


def sub_round(l, f, i):
    return f * l['nb'] * e(l['n'][i]) + 3.0 + US * 1e-6 * f * l['N'] + 3.5   # extraction, clustering floor + rate, postprocess_templates


rows = {}
for name, l in L.items():
    seed = l['univ'] + l['clu1']
    r = {'baseline_4round': l['total4'], 'conventional_2round': l['total2'], 'floor_free_dictionary': base(l) + final(l)}
    for f in (1/2, 1/4, 1/6, 1/8, 1/16):
        r[f'D3_f={f:.3f}'] = base(l) + seed + sub_round(l, f, 0) + sub_round(l, f, 1) + final(l)
        r[f'D3s_f={f:.3f}_seed_too'] = base(l) + f * l['univ'] + US * 1.3e-6 * f * l['nuniv'] + sub_round(l, f, 0) + sub_round(l, f, 1) + final(l)
    n = l['n'][-1]
    for P in (0.25, 0.5, 1, 2, 5):
        r[f'D2_seed_full_P={P}'] = base(l) + seed + P * l['nb'] * online(n, birth=0, merge=0) + final(l)
        r[f'D1_P={P}'] = base(l) + P * l['nb'] * online(n) + final(l)
        r[f'D1_noclu_P={P}'] = base(l) + P * l['nb'] * online(n) + l['ext'][-1] + l['merge'] + l['export']
    rows[name] = r
    print(name, 'per-round', round(l['ext'][-1] + l['clu'][-1]), 's; online s/batch R=1,10,100:', [round(online(n, R), 3) for R in (1, 10, 100)])
    for k, v in r.items():
        print(f'  {k:32s} {v:7.0f} s {v/60:6.1f} min  {100*(1-v/l["total4"]):5.1f}% below 4-round')
(OUT / 'cost_model.json').write_text(json.dumps(rows, indent=1))
