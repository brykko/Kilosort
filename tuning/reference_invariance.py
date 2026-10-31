"""Reference-invariance check (spec §8.3): do comparisons between sorts depend on which
reference sort theta_ref was built from?

Scores every sort against every reference and reports the pairwise differences in S under each
reference. A sort scored against a reference built from itself also gains the self-leak (§8.4),
which shows up as the diagonal advantage. Every sort is cleaned first (spec §6.1-6.3; cleaning.npz
in each sort folder).
"""
import argparse
import itertools
import json

import numpy as np

from score import score_sort
from cleaning import load_clean_units

p = argparse.ArgumentParser()
p.add_argument('--sorts', nargs='+', required=True, help='name=folder')
p.add_argument('--refs', nargs='+', required=True, help='name=theta_ref.npz')
p.add_argument('--t0', type=float, required=True)
p.add_argument('--duration', type=float, required=True)
p.add_argument('--raw-artefacts', required=True)
p.add_argument('--out', required=True)
args = p.parse_args()

sorts = dict(a.split('=', 1) for a in args.sorts)
refs = {k: np.load(v) for k, v in (a.split('=', 1) for a in args.refs)}
clean = {s: load_clean_units(folder, args.raw_artefacts) for s, folder in sorts.items()}
S = {s: {r: score_sort(clean[s][0], ref, args.t0, args.duration, clean[s][2])[0] for r, ref in refs.items()}
     for s in sorts}
pairs = {f'{b} - {a}': {r: S[b][r] - S[a][r] for r in refs} for a, b in itertools.combinations(sorts, 2)}
Path = __import__('pathlib').Path
Path(args.out).write_text(json.dumps({'S': S, 'pairwise_delta': pairs}, indent=2))

print('S (bits/s): rows = sort, columns = reference')
print(f"{'':28s}" + ''.join(f'{r:>14s}' for r in refs))
for s in sorts:
    print(f'{s:28s}' + ''.join(f'{S[s][r]:14.1f}' for r in refs))
print('\nPairwise differences under each reference')
for name, d in pairs.items():
    v = list(d.values())
    print(f'{name:45s}' + ''.join(f'{x:+10.1f}' for x in v) + f'   spread {max(v) - min(v):5.1f}')
