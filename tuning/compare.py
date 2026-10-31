"""Compare a candidate sort with default (spec §7): guards G1-G3 and G5-G9 from evaluation.json
files, and the matched-unit report (§7.6). G4 applies when the sorts were evaluated with hybrid truth.

Default is one or more replicate sorts (each with evaluation.json). Each guard's quantity is
compared with the replicate mean; tolerances tau are 2 x the SD across replicates (§7), so with a
single replicate they are unknown and the guards that need them report 'no tau'.

Matched units (§7.6): eligible candidate units are matched to eligible units of the first default
replicate by spike-train agreement, matched / (n_a + n_b - matched) with +-0.4 ms tolerance, among
units whose median depths are <= 100 um apart; a pair matches if its agreement is >= 0.5.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cleaning import load_clean_units

TOL_S = 0.4e-3
MATCH_UM = 100.0
MATCH_MIN = 0.5


def agreement(ta, tb):
    """Matched spikes / (n_a + n_b - matched), each a-spike matching at most one b-spike nearby."""
    i = np.clip(np.searchsorted(tb, ta), 1, tb.size - 1)
    near = np.minimum(np.abs(tb[i] - ta), np.abs(tb[i - 1] - ta)) <= TOL_S
    m = min(int(near.sum()), tb.size)
    return m / (ta.size + tb.size - m)


def match_units(cand, default, raw_artefacts):
    """Best default unit and agreement for each eligible candidate unit."""
    out = {}
    (ca, cd, _), (da, dd, _) = (load_clean_units(e['sort'], raw_artefacts) for e in (cand, default))
    d_units = [int(c) for c in default['S_i']]
    for c in (int(c) for c in cand['S_i']):
        best = (None, 0.0)
        ta = np.sort(ca[c])
        for d in d_units:
            if abs(cd[c] - dd[d]) <= MATCH_UM:
                a = agreement(ta, np.sort(da[d]))
                if a > best[1]:
                    best = (d, a)
        out[c] = best
    return out


def matched_report(cand, default, matches):
    pairs = {c: d for c, (d, a) in matches.items() if a >= MATCH_MIN}
    hit_d = set(pairs.values())
    un_c = [v for c, v in cand['S_i'].items() if int(c) not in pairs]
    un_d = [v for d, v in default['S_i'].items() if int(d) not in hit_d]
    summary = lambda v: {'n': len(v), 'sum': float(np.sum(v)), 'median': float(np.median(v)) if v else None}
    return {'n_matched': len(pairs),
            'S_matched_candidate': float(sum(cand['S_i'][str(c)] for c in pairs)),
            'S_matched_default': float(sum(default['S_i'][str(d)] for d in hit_d)),
            'unmatched_candidate': summary(un_c), 'unmatched_default': summary(un_d)}


def guards(cand, defaults):
    def check(name, value, default_value, ok, tau=False):
        """ok(value, default, tau); tau=False for guards without a tolerance, None if it's unknown."""
        status = 'no tau' if tau is None else ('pass' if ok(value, default_value, tau) else 'FAIL')
        return {'guard': name, 'candidate': value, 'default': default_value,
                'tau': tau if tau is not False else None, 'status': status}

    def relative(name, key, ok):
        v = np.array([e[key] for e in defaults], float)
        tau = float(2 * v.std(ddof=1)) if v.size > 1 else None
        return check(name, cand[key], float(v.mean()), ok, tau)

    def hybrid_guards():
        if not cand.get('hybrid') or not all(e.get('hybrid') for e in defaults):
            return []
        m = np.array([e['hybrid']['median_agreement'] for e in defaults])
        tau = float(2 * m.std(ddof=1)) if m.size > 1 else None
        rec = float(np.mean([e['hybrid']['n_recovered'] for e in defaults]))
        return [check('G4 hybrid median agreement', cand['hybrid']['median_agreement'], float(m.mean()),
                      lambda v, d, t: v >= d - t, tau),
                check('G4 hybrid units recovered', cand['hybrid']['n_recovered'], rec, lambda v, d, t: v >= d - 1),
                check('G4 hybrid HD pairs merged', len(cand['hybrid']['merged_pairs']), 0, lambda v, d, t: v == 0)]

    vals = lambda e: np.array(list(e['S_i'].values()))
    q = [np.percentile(vals(e), 25) for e in defaults]
    at_most = lambda v, d, t: v <= d + t
    return [
        check('G1 eligible units', cand['n_eligible'], float(np.mean([e['n_eligible'] for e in defaults])),
              lambda v, d, t: 0.8 * d <= v <= 1.25 * d),
        relative('G2 S_track (backstop)', 'S_track', lambda v, d, t: v >= d - 5 * t),
        relative('G3 suspected splits', 'n_suspected_splits', at_most),
        *hybrid_guards(),
        relative('G5 near-noise share of S', 'S_share_near_noise', at_most),
        relative('G6 spikes removed by cleaning', 'fraction_spikes_removed', at_most),
        relative('G6 flagged duplicate pairs', 'n_flagged_pairs', at_most),
        # G7: candidate units below the 25th percentile of default's S_i, against default's own count
        check('G7 untuned units', float(np.mean([np.sum(vals(cand) < qq) for qq in q])),
              float(np.mean([np.sum(vals(e) < qq) for e, qq in zip(defaults, q)])), lambda v, d, t: v >= 0.9 * d),
        check('G8 raw mask hash', cand['raw_artefacts_sha256'], defaults[0]['raw_artefacts_sha256'],
              lambda v, d, t: v == d),
        relative('G9 calibration C (backstop)', 'C', lambda v, d, t: v >= d - 5 * t),
    ]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--candidate', required=True, help='sort folder with evaluation.json')
    p.add_argument('--default', nargs='+', required=True, help='default replicate sort folders')
    p.add_argument('--raw-artefacts', required=True)
    p.add_argument('--out', help='output .json (default: comparison.json in the candidate folder)')
    args = p.parse_args()
    load = lambda f: json.loads((Path(f) / 'evaluation.json').read_text())
    cand, defaults = load(args.candidate), [load(f) for f in args.default]
    if len({e['reference_sha256'] for e in [cand, *defaults]}) > 1:
        raise SystemExit('candidate and default were scored against different references')
    g = guards(cand, defaults)
    rep = matched_report(cand, defaults[0], match_units(cand, defaults[0], args.raw_artefacts))
    S_def = float(np.mean([e['S'] for e in defaults]))
    result = {'candidate': args.candidate, 'default': args.default, 'S': cand['S'], 'S_default': S_def,
              'dS': cand['S'] - S_def, 'guards': g, 'matched_units': rep}
    Path(args.out or Path(args.candidate) / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(f"S {cand['S']:.1f} vs default {S_def:.1f} (dS {cand['S'] - S_def:+.1f} bits/s)")
    for x in g:
        val = x['candidate'] if isinstance(x['candidate'], str) else f"{x['candidate']:.4g}"
        dv = x['default'] if isinstance(x['default'], str) else f"{x['default']:.4g}"
        print(f"  {x['guard']:32s} {val[:12]:>12s}  default {dv[:12]:>12s}  {x['status']}")
    print('matched units:', json.dumps(rep))
