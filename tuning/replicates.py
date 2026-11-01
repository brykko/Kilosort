"""Phase 0 item 8.1 (spec §8): spread of default replicates. From each replicate's evaluation.json,
reports S and every guard quantity with its mean, SD and tau = 2 x SD (§7), and the comparison margin
delta = max(1% of S, 2 sigma_S) (§5.4). Writes the summary to --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np

QUANTITIES = ['S', 'S_track', 'n_eligible', 'S_share_near_noise', 'fraction_spikes_removed', 'n_flagged_pairs',
              'masked_time_s']
HYBRID = ['median_agreement', 'n_recovered']

p = argparse.ArgumentParser()
p.add_argument('--sorts', nargs='+', required=True, help='replicate sort folders')
p.add_argument('--out', required=True)
args = p.parse_args()

ev = [json.loads((Path(s) / 'evaluation.json').read_text()) for s in args.sorts]
if len({e['reference_sha256'] for e in ev}) > 1:
    raise SystemExit('replicates were scored against different references')
rows = {q: [e[q] for e in ev] for q in QUANTITIES}
if all(e.get('hybrid') for e in ev):
    rows.update({f'hybrid_{q}': [e['hybrid'][q] for e in ev] for q in HYBRID})
    rows['hybrid_merged_pairs'] = [len(e['hybrid']['merged_pairs']) for e in ev]
stats = {q: {'values': v, 'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)),
             'tau': float(2 * np.std(v, ddof=1))} for q, v in rows.items()}
S = stats['S']
summary = {'sorts': args.sorts, 'n': len(ev), 'stats': stats,
           'sigma_S': S['sd'], 'delta': max(0.01 * S['mean'], 2 * S['sd'])}
Path(args.out).write_text(json.dumps(summary, indent=2))
print(f"{'quantity':28s} {'mean':>10s} {'SD':>9s} {'tau':>9s}   values")
for q, s in stats.items():
    print(f"{q:28s} {s['mean']:10.4g} {s['sd']:9.3g} {s['tau']:9.3g}   " + ' '.join(f'{x:.4g}' for x in s['values']))
print(f"\nsigma_S = {S['sd']:.2f} bits/s; delta = max(1% of S, 2 sigma_S) = {summary['delta']:.2f} bits/s")
