"""Tables and paired comparisons from the main, vari, sens and scale stages. Writes OUT/results.md and prints it."""
import glob
import json
import os
import sys

import numpy as np

OUT = r'C:\data\ks4_tuning\exploratory\interleaving\simulation'
VARS = ('stationary', 'drift', 'nonstat')
rng = np.random.default_rng(0)
out = []


def P(s=''):
    out.append(s)


def load(stage):
    res = []
    for f in sorted(glob.glob(os.path.join(OUT, stage, '*.json'))):
        res.append(json.load(open(f)))
    return res


def boot(x, n=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    i = rng.integers(0, len(x), (n, len(x)))
    m = x[i].mean(1)
    return (x.mean(), *np.percentile(m, [2.5, 97.5]))


def fmt(c, d=3):
    return f'{c[0]:+.{d}f} [{c[1]:+.{d}f}, {c[2]:+.{d}f}]'


# ---- collect final records: E[(variant, seed, output)] = dict
main = load('main')
E = {}
TRUTH = {}
CURVES = {}
for d in main:
    v, s = d['job']['variant'], d['job']['seed']
    TRUTH[(v, s)] = d['truth']
    for a in d['arms']:
        recs = a['records']
        name = a['name']
        if name == 'R4' or name == 'Rd3':
            pre = 'R' if name == 'R4' else 'Rd'
            for r in recs:
                if r['tag'] == 'round1':
                    E[(v, s, pre + '1')] = r
                else:
                    E[(v, s, pre + r['tag'][-1])] = r
        else:
            fin = [r for r in recs if r['tag'] == 'final'][-1]
            E[(v, s, name)] = fin
            CURVES[(v, s, name)] = [(float(r['tag'][2:]), r['te_med'], r['te_mean']) for r in recs if r['tag'].startswith('it')]

seeds = sorted({k[1] for k in E})
SHORT = {'I1_shuffled': 'I(1)', 'I1_time': 'I(1) time', 'I0.25_shuffled': 'I(0.25)', 'S0.25_3': 'S', 'O': 'O',
         'I1_shuffled_track': 'I(1)+track', 'I1_time_track': 'I(1) time+track', 'Tr3': 'R3+track'}


def get(v, s, arm, lab, key):
    r = E.get((v, s, arm))
    if r is None:
        return np.nan
    if lab is None:
        return r.get(key, np.nan)
    return r[lab][key]


def row(v, arm, lab, keys):
    cells = []
    for k in keys:
        x = [get(v, s, arm, lab, k) for s in seeds]
        x = np.array(x, float)
        cells.append(f'{np.nanmean(x):.3f}')
    return cells


OUTPUTS = [('O', 'T'), ('O', 'C'), ('R2', 'T'), ('R2', 'C'), ('R3', 'C'), ('R4', 'T'), ('R4', 'C'), ('Rd2', 'C'), ('Rd3', 'C'), ('S0.25_3', 'C'),
           ('S0.25_3', 'T'), ('I0.25_shuffled', 'T'), ('I0.25_shuffled', 'C'), ('I1_shuffled', 'T'), ('I1_shuffled', 'C'),
           ('I1_time', 'T'), ('I1_time', 'C'), ('I1_shuffled_track', 'T'), ('I1_shuffled_track', 'C'), ('I1_time_track', 'T'), ('I1_time_track', 'C'), ('Tr3', 'T'), ('Tr3', 'C')]

for v in VARS:
    P(f'\n### {v}: means over seeds {seeds[0]}-{seeds[-1]}\n')
    keys = ['acc', 'found', 'rec', 'prec', 'n_out', 'n_false', 'split', 'merged', 'pair_res', 'found_poor', 'found_reg', 'found_lt05', 'acc_gt15']
    P('| output | FPE | K | ' + ' | '.join(keys) + ' | te med | te mean |')
    P('|---|' + '---|' * (len(keys) + 4))
    for arm, lab in OUTPUTS:
        if (v, seeds[0], arm) not in E:
            continue
        fp = np.nanmean([get(v, s, arm, None, 'fpe') for s in seeds])
        K = np.nanmean([get(v, s, arm, None, 'K') for s in seeds])
        te = np.nanmean([get(v, s, arm, None, 'te_med') for s in seeds]); tm = np.nanmean([get(v, s, arm, None, 'te_mean') for s in seeds])
        P(f'| {SHORT.get(arm, arm)} {lab} | {fp:.2f} | {K:.0f} | ' + ' | '.join(row(v, arm, lab, keys)) + f' | {te:.3f} | {tm:.3f} |')


def diff(v, a, la, b, lb, key):
    return np.array([get(v, s, a, la, key) - get(v, s, b, lb, key) for s in seeds])


# ---- H1
P('\n### H1: dictionary quality against cost\n')
P('Template error at the end of each learning stage (median / mean over units, mean over seeds; FPE counts learning passes).\n')
P('| variant | R1 (1) | R2 (2) | R3 (3) | R4 (4) | Rd2 | I at 0.25 pass of I(1) | I at 0.5 | I at 1.0 | I(0.25) | oracle |')
P('|---|---|---|---|---|---|---|---|---|---|---|')


def curve_at(v, s, name, x, idx):
    c = CURVES.get((v, s, name))
    if not c:
        return np.nan
    c = np.array(c)
    return c[np.argmin(np.abs(c[:, 0] - x)), idx]


for v in VARS:
    cells = []
    for arm in ('R1', 'R2', 'R3', 'R4', 'Rd2'):
        a = np.nanmean([get(v, s, arm, None, 'te_med') for s in seeds]); b = np.nanmean([get(v, s, arm, None, 'te_mean') for s in seeds])
        cells.append(f'{a:.3f}/{b:.3f}')
    for x in (.25, .5):
        a = np.nanmean([curve_at(v, s, 'I1_shuffled', x, 1) for s in seeds]); b = np.nanmean([curve_at(v, s, 'I1_shuffled', x, 2) for s in seeds])
        cells.append(f'{a:.3f}/{b:.3f}')
    for arm in ('I1_shuffled', 'I0.25_shuffled', 'O'):
        a = np.nanmean([get(v, s, arm, None, 'te_med') for s in seeds]); b = np.nanmean([get(v, s, arm, None, 'te_mean') for s in seeds])
        cells.append(f'{a:.3f}/{b:.3f}')
    P(f'| {v} | ' + ' | '.join(cells) + ' |')
P('\nPaired differences in template error, I minus target (negative: I better), 95% bootstrap CI over seeds:\n')
P('| variant | measure | I(1) − R1 | I(1) − R2 | I(1) − R3 | I(1) − R4 | I(0.25) − R2 | I(0.25) − R3 | at 0.5 pass of I(1) − R2 | S learned? |')
P('|---|---|---|---|---|---|---|---|---|---|')
for v in VARS:
    for key, idx in (('te_med', 1), ('te_mean', 2)):
        cells = [fmt(boot(diff(v, 'I1_shuffled', None, t, None, key))) for t in ('R1', 'R2', 'R3', 'R4')]
        cells += [fmt(boot(diff(v, 'I0.25_shuffled', None, t, None, key))) for t in ('R2', 'R3')]
        cells += [fmt(boot([curve_at(v, s, 'I1_shuffled', .5, idx) - get(v, s, 'R2', None, key) for s in seeds]))]
        cells += [fmt(boot(diff(v, 'S0.25_3', None, 'R2', None, key)))]
        P(f'| {v} | {key} | ' + ' | '.join(cells) + ' |')
P('\nFirst pass fraction at which I(1) reaches a target (seeds reaching it of 10, median fraction):\n')
for v in VARS:
    for key, idx in (('te_med', 1), ('te_mean', 2)):
        ln = []
        for t in ('R1', 'R2', 'R3'):
            hit = []
            for s in seeds:
                c = np.array(CURVES[(v, s, 'I1_shuffled')])
                tgt = get(v, s, t, None, key)
                j = np.flatnonzero(c[:, idx] <= tgt)
                hit.append(c[j[0], 0] if len(j) else np.nan)
            ln.append(f'{t}: {np.isfinite(hit).sum()}/10, median {np.nanmedian(hit) if np.isfinite(hit).any() else float("nan"):.2f}')
        P(f'- {v}, {key}: ' + '; '.join(ln))

# ---- compute
P('\n### Compute\n')
P('Matched-filter ops (10⁹ template-samples) and clock (s, one core, machine under load from other jobs), mean over seeds.\n')
P('| arm | stationary ops | clock | drift ops | clock |')
P('|---|---|---|---|---|')
for arm in ('R2', 'R3', 'R4', 'Rd3', 'S0.25_3', 'I0.25_shuffled', 'I1_shuffled', 'I1_time'):
    cells = []
    for v in ('stationary', 'drift'):
        cells.append(f"{np.nanmean([get(v, s, arm, None, 'mf_ops') for s in seeds]) / 1e9:.1f}")
        cells.append(f"{np.nanmean([get(v, s, arm, None, 'clock') for s in seeds]):.0f}")
    P(f'| {SHORT.get(arm, arm)} | ' + ' | '.join(cells) + ' |')

# ---- primary pairs
P('\n### Paired differences on mean accuracy and found rate (95% CI over seeds)\n')
P('| variant | comparison | Δ accuracy | Δ found | Δ precision | Δ recall | Δ false units | Δ merged | Δ split |')
P('|---|---|---|---|---|---|---|---|---|')
PAIRS = [('I1_shuffled', 'T', 'R2', 'C'), ('I1_shuffled', 'T', 'R3', 'C'), ('I1_shuffled', 'T', 'R4', 'C'), ('I1_shuffled', 'C', 'R2', 'C'), ('I1_shuffled', 'C', 'R3', 'C'), ('I1_shuffled', 'C', 'R4', 'C'),
         ('I0.25_shuffled', 'C', 'R2', 'C'), ('I0.25_shuffled', 'C', 'R3', 'C'), ('S0.25_3', 'C', 'R2', 'C'), ('S0.25_3', 'C', 'R3', 'C'), ('S0.25_3', 'C', 'I1_shuffled', 'C'), ('S0.25_3', 'C', 'I1_shuffled', 'T'),
         ('I1_time', 'C', 'I1_shuffled', 'C'), ('I1_time', 'T', 'I1_shuffled', 'T'), ('R3', 'C', 'R2', 'C'), ('R4', 'C', 'R2', 'C'), ('Rd3', 'C', 'R3', 'C'), ('I1_shuffled', 'C', 'Rd3', 'C')]
for v in VARS:
    for a, la, b, lb in PAIRS:
        cells = [fmt(boot(diff(v, a, la, b, lb, k)), 2 if k == 'n_false' else 3) for k in ('acc', 'found', 'prec', 'rec', 'n_false', 'merged', 'split')]
        P(f'| {v} | {SHORT.get(a, a)} {la} − {SHORT.get(b, b)} {lb} | ' + ' | '.join(cells) + ' |')

# ---- H2: per-unit on poor-match units
P('\n### H2: poor-match units\n')


def units(v, s, arm, lab, mask_fn, field='found'):
    r = E.get((v, s, arm))
    if r is None:
        return np.nan
    t = TRUTH[(v, s)]
    f = np.array(r[f'{lab}_{field}'], float)
    m = mask_fn(t, s, v)
    return f[m].mean() if m.any() else np.nan


def poor(t, s, v):
    return np.array(t['um']) < .75


def poor_det(t, s, v):
    """Poor units with enough universal score to be seen at all: norm x um >= 8 (one spike at mean amplitude crosses Th_u, with margin)."""
    return (np.array(t['um']) < .75) & (np.array(t['norm']) * np.array(t['um']) >= 8)


def poor_und(t, s, v):
    return (np.array(t['um']) < .75) & (np.array(t['norm']) * np.array(t['um']) < 8)


def kind(k):
    return lambda t, s, v: (np.array(t['kinds']) == k) & (np.array(t['um']) < .75)


def reg(t, s, v):
    return np.array(t['um']) >= .75


MASKS = [('all poor', poor), ('poor, score ≥ 8', poor_det), ('poor, score < 8', poor_und), ('triphasic', kind('triphasic')), ('axonal', kind('axonal')), ('wide', kind('wide')), ('regular', reg)]
counts = {n: np.mean([m(TRUTH[(v, s)], s, v).sum() for v in VARS for s in seeds]) for n, m in MASKS}
P('Units per recording (mean): ' + ', '.join(f'{n} {c:.1f}' for n, c in counts.items()) + '\n')
P('Found rate, mean over seeds:\n')
ARMS_H2 = [('O', 'T'), ('R2', 'C'), ('R3', 'C'), ('R4', 'C'), ('Rd2', 'C'), ('Rd3', 'C'), ('S0.25_3', 'C'), ('I1_shuffled', 'T'), ('I1_shuffled', 'C'), ('I1_time', 'T'), ('I0.25_shuffled', 'T')]
P('| variant | mask | ' + ' | '.join(f'{SHORT.get(a, a)} {l}' for a, l in ARMS_H2) + ' |')
P('|---|---|' + '---|' * len(ARMS_H2))
for v in VARS:
    for n, m in MASKS:
        cells = [f'{np.nanmean([units(v, s, a, l, m) for s in seeds]):.2f}' for a, l in ARMS_H2]
        P(f'| {v} | {n} | ' + ' | '.join(cells) + ' |')
P('\nPaired differences in found rate on all poor units (and on those with score ≥ 8):\n')
P('| variant | I(1) T − R3 C | I(1) C − R3 C | I(1) C − R4 C | Rd3 C − R3 C | I(1) C − Rd3 C | R4 C − R2 C | R3 C − R2 C |')
P('|---|---|---|---|---|---|---|---|')
for v in VARS:
    for mn, m in (('all', poor), ('≥8', poor_det)):
        def d(a, la, b, lb):
            return fmt(boot([units(v, s, a, la, m) - units(v, s, b, lb, m) for s in seeds]), 2)
        P(f"| {v} {mn} | {d('I1_shuffled','T','R3','C')} | {d('I1_shuffled','C','R3','C')} | {d('I1_shuffled','C','R4','C')} | {d('Rd3','C','R3','C')} | {d('I1_shuffled','C','Rd3','C')} | {d('R4','C','R2','C')} | {d('R3','C','R2','C')} |")

# ---- strata
P('\n### Accuracy by rate and poor-match status (mean over units and seeds)\n')
RATE = [('<0.5 Hz', 0, .5), ('0.5-3', .5, 3), ('3-15', 3, 15), ('>15', 15, 1e9)]
P('| variant | stratum | n units | R3 C | I(1) T | I(1) C | I(1) time T | S C | O T |')
P('|---|---|---|---|---|---|---|---|---|')
for v in VARS:
    for pn, pf in (('reg', False), ('poor', True)):
        for rn, lo, hi in RATE:
            def mk(t, s, v_):
                r = np.array(t['rates'])
                return ((np.array(t['um']) < .75) == pf) & (r >= lo) & (r < hi)
            n = np.mean([mk(TRUTH[(v, s)], s, v).sum() for s in seeds])
            cells = [f"{np.nanmean([units(v, s, a, l, mk, 'acc') for s in seeds]):.2f}" for a, l in (('R3', 'C'), ('I1_shuffled', 'T'), ('I1_shuffled', 'C'), ('I1_time', 'T'), ('S0.25_3', 'C'), ('O', 'T'))]
            P(f'| {v} | {pn} {rn} | {n:.1f} | ' + ' | '.join(cells) + ' |')
P('\nPer-unit accuracy distribution, all units pooled over seeds (percentiles 10/25/50/75/90):\n')
P('| variant | output | ' + ' | '.join(['p10', 'p25', 'p50', 'p75', 'p90']) + ' |')
P('|---|---|---|---|---|---|---|')
for v in VARS:
    for a, l in (('O', 'T'), ('R2', 'C'), ('R3', 'C'), ('R4', 'C'), ('Rd3', 'C'), ('S0.25_3', 'C'), ('I1_shuffled', 'T'), ('I1_shuffled', 'C'), ('I1_time', 'T')):
        x = np.concatenate([E[(v, s, a)][f'{l}_acc'] for s in seeds if (v, s, a) in E])
        P(f'| {v} | {SHORT.get(a, a)} {l} | ' + ' | '.join(f'{q:.2f}' for q in np.percentile(x, [10, 25, 50, 75, 90])) + ' |')

# ---- H3
P('\n### H3: sparse and silent units (nonstat), found rate\n')
P('| variant | group | n | R3 C | I(1) T | I(1) C | I(1) time T | I(1) time C | Δ I(1) C − R3 C | Δ I(1) T − R3 C | Δ I(1) time T − R3 C |')
P('|---|---|---|---|---|---|---|---|---|---|---|')


def sparse(t, s, v):
    return np.array(t['rates']) < .5


def silent(t, s, v):
    return np.array(t['silent']) > 0


def burst(t, s, v):
    return np.array(t['burst']) > 0


for v in VARS:
    for gn, m in (('< 0.5 Hz', sparse),) + ((('silent epoch', silent), ('bursting', burst)) if v == 'nonstat' else ()):
        n = np.mean([m(TRUTH[(v, s)], s, v).sum() for s in seeds])
        vals = [f'{np.nanmean([units(v, s, a, l, m) for s in seeds]):.2f}' for a, l in (('R3', 'C'), ('I1_shuffled', 'T'), ('I1_shuffled', 'C'), ('I1_time', 'T'), ('I1_time', 'C'))]
        ds = [fmt(boot([units(v, s, a, l, m) - units(v, s, 'R3', 'C', m) for s in seeds]), 2) for a, l in (('I1_shuffled', 'C'), ('I1_shuffled', 'T'), ('I1_time', 'T'))]
        P(f'| {v} | {gn} | {n:.1f} | ' + ' | '.join(vals) + ' | ' + ' | '.join(ds) + ' |')

# ---- H5
P('\n### H5: drift\n')
P('| comparison | Δ accuracy, drift | Δ accuracy, stationary | Δ found, drift | Δ found, stationary |')
P('|---|---|---|---|---|')
for a, la, b, lb in [('I1_shuffled', 'T', 'R3', 'C'), ('I1_shuffled', 'C', 'R3', 'C'), ('I1_time', 'T', 'R3', 'C'), ('I1_time', 'C', 'R3', 'C'), ('I1_time', 'C', 'I1_shuffled', 'C')]:
    P(f"| {SHORT.get(a, a)} {la} − {SHORT.get(b, b)} {lb} | {fmt(boot(diff('drift', a, la, b, lb, 'acc')))} | {fmt(boot(diff('stationary', a, la, b, lb, 'acc')))} | {fmt(boot(diff('drift', a, la, b, lb, 'found')))} | {fmt(boot(diff('stationary', a, la, b, lb, 'found')))} |")
for a, la, b, lb in [('I1_shuffled_track', 'T', 'R3', 'C'), ('I1_shuffled_track', 'C', 'R3', 'C'), ('Tr3', 'C', 'R3', 'C'), ('Tr3', 'T', 'R3', 'C'), ('I1_shuffled_track', 'C', 'Tr3', 'C'), ('I1_shuffled_track', 'T', 'Tr3', 'T')]:
    P(f"| {SHORT.get(a, a)} {la} − {SHORT.get(b, b)} {lb} | {fmt(boot(diff('drift', a, la, b, lb, 'acc')))} | n/a | {fmt(boot(diff('drift', a, la, b, lb, 'found')))} | n/a |")

# ---- H6
P('\n### H6: near-identical pairs, resolved fraction (mean over seeds)\n')
P('| variant | O T | R2 C | R3 C | R4 C | Rd3 C | S C | I(1) T | I(1) C | I(1) time T | Δ R3 C − I(1) T | Δ R3 C − I(1) C |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|')
for v in VARS:
    cells = [f"{np.nanmean([get(v, s, a, l, 'pair_res') for s in seeds]):.2f}" for a, l in (('O', 'T'), ('R2', 'C'), ('R3', 'C'), ('R4', 'C'), ('Rd3', 'C'), ('S0.25_3', 'C'), ('I1_shuffled', 'T'), ('I1_shuffled', 'C'), ('I1_time', 'T'))]
    P(f"| {v} | " + ' | '.join(cells) + f" | {fmt(boot(diff(v, 'R3', 'C', 'I1_shuffled', 'T', 'pair_res')), 2)} | {fmt(boot(diff(v, 'R3', 'C', 'I1_shuffled', 'C', 'pair_res')), 2)} |")

# ---- vari
vari = load('vari')
if vari:
    P('\n### Run-to-run variability (stationary, data seeds 1-3, algorithm seeds 0-2)\n')
    V = {}
    for d in vari:
        s, alg = d['job']['seed'], d['job']['alg']
        for a in d['arms']:
            nm = a['name'].split('_')[0]
            r = a['records'][-1]
            V[(s, alg, nm)] = r
    for s in (1, 2, 3):
        V[(s, 0, 'R3')] = E[('stationary', s, 'R3')]
        V[(s, 0, 'I1')] = E[('stationary', s, 'I1_shuffled')]
    P('| arm, output | sd of mean accuracy across algorithm seeds | mean Jaccard of found-unit sets | te med sd |')
    P('|---|---|---|---|')
    for nm, lab in (('R3', 'C'), ('I1', 'T'), ('I1', 'C')):
        sds, jac, tes = [], [], []
        for s in (1, 2, 3):
            ac = [V[(s, a, nm)][lab]['acc'] for a in (0, 1, 2)]
            sds.append(np.std(ac, ddof=1))
            tes.append(np.std([V[(s, a, nm)]['te_med'] for a in (0, 1, 2)], ddof=1))
            F = [np.array(V[(s, a, nm)][f'{lab}_found'], bool) for a in (0, 1, 2)]
            for i, j in ((0, 1), (0, 2), (1, 2)):
                jac.append((F[i] & F[j]).sum() / max(1, (F[i] | F[j]).sum()))
        P(f'| {nm} {lab} | {np.mean(sds):.4f} | {np.mean(jac):.3f} | {np.mean(tes):.4f} |')
    P(f"\nBetween-data-seed sd of mean accuracy for scale: R3 C {np.std([E[('stationary', s, 'R3')]['C']['acc'] for s in seeds], ddof=1):.3f}.")

# ---- sens
sens = load('sens')
if sens:
    P('\n### Sensitivity of I (seeds 1-2; T and C mean accuracy, found, false, pairs; change from tuned I on the same seeds)\n')
    P('| variant | config | arm | acc T | Δ T | acc C | Δ C | found T | false T | K | te med |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    rows = {}
    for d in sens:
        v, s = d['job']['variant'], d['job']['seed']
        for a in d['arms']:
            r = [x for x in a['records'] if x['tag'] == 'final'][-1]
            rows.setdefault((v, a['name'], a['arm']), []).append((s, r))
    for (v, nm, arm), lst in sorted(rows.items()):
        base = [E[(v, s, arm)] for s, _ in lst]
        at = np.mean([r['T']['acc'] for _, r in lst]); ac = np.mean([r['C']['acc'] for _, r in lst])
        bt = np.mean([b['T']['acc'] for b in base]); bc = np.mean([b['C']['acc'] for b in base])
        P(f"| {v} | {nm} | {SHORT.get(arm, arm)} | {at:.3f} | {at - bt:+.3f} | {ac:.3f} | {ac - bc:+.3f} | {np.mean([r['T']['found'] for _, r in lst]):.2f} | {np.mean([r['T']['n_false'] for _, r in lst]):.1f} | {np.mean([r['K'] for _, r in lst]):.0f} | {np.mean([r['te_med'] for _, r in lst]):.3f} |")

# ---- scale
scale = load('scale')
if scale:
    P('\n### Scale check (seeds 1-3, stationary, 120 s unless stated; base = main run, same seeds)\n')
    P('| condition | units/chan | median spikes/unit/batch | R3 C acc | I(1) T acc | I(1) C acc | I(1) C − R3 C per seed | I(1) T − R3 C per seed | found: R3 / I(1) T / I(1) C | sparse found: R3 / I(1) C | pair: R3 / I(1) C | false: R3 / I(1) T / I(1) C | te med: R3 / I(1) |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    S = {}
    for d in scale:
        for a in d['arms']:
            cond, arm = a['name'].split('|')
            r = [x for x in a['records'] if x['tag'] in ('final', 'round3')][-1]
            S.setdefault(cond, {}).setdefault(arm, {})[d['job']['seed']] = (r, d)
    base = {'R3': {}, 'I1_shuffled': {}}
    for s_ in (1, 2, 3):
        dj = dict(job=dict(over=dict(T=120.0)), truth=TRUTH[('stationary', s_)])
        base['R3'][s_] = (E[('stationary', s_, 'R3')], dj)
        base['I1_shuffled'][s_] = (E[('stationary', s_, 'I1_shuffled')], dj)
    S['base'] = base
    for cond in ('base', 'u24', 'u96', 'u192', 'rate1', 'rate15', 'batch0.5', 'batch8', 'T360'):
        if cond not in S:
            continue
        c = S[cond]
        sds = sorted(set(c.get('R3', {})) & set(c.get('I1_shuffled', {})))
        if not sds:
            continue
        def m(arm, lab, k):
            return np.nanmean([c[arm][s_][0][lab][k] for s_ in sds])
        over = c['R3'][sds[0]][1]['job']['over']
        nu = over.get('n_units', 48); bs = over.get('batch_s', 2.0)
        ms = np.median(c['R3'][sds[0]][1]['truth']['n_spikes']) / (over.get('T', 120) / bs)
        dd = ' '.join(f"{c['I1_shuffled'][s_][0]['C']['acc'] - c['R3'][s_][0]['C']['acc']:+.3f}" for s_ in sds)
        dt = ' '.join(f"{c['I1_shuffled'][s_][0]['T']['acc'] - c['R3'][s_][0]['C']['acc']:+.3f}" for s_ in sds)
        te = [np.nanmean([c[a][s_][0]['te_med'] for s_ in sds]) for a in ('R3', 'I1_shuffled')]
        P(f"| {cond} | {nu / 32:.2f} | {ms:.1f} | {m('R3','C','acc'):.3f} | {m('I1_shuffled','T','acc'):.3f} | {m('I1_shuffled','C','acc'):.3f} | {dd} | {dt} | {m('R3','C','found'):.2f} / {m('I1_shuffled','T','found'):.2f} / {m('I1_shuffled','C','found'):.2f} | {m('R3','C','found_lt05'):.2f} / {m('I1_shuffled','C','found_lt05'):.2f} | {m('R3','C','pair_res'):.2f} / {m('I1_shuffled','C','pair_res'):.2f} | {m('R3','C','n_false'):.1f} / {m('I1_shuffled','T','n_false'):.1f} / {m('I1_shuffled','C','n_false'):.1f} | {te[0]:.3f} / {te[1]:.3f} |")
    if 'drift10' in S:
        P('\n### Mild drift (10 µm peak to peak, seeds 1-3)\n')
        P('| arm | T acc | C acc | found T | found C | false T | false C | per seed C acc |')
        P('|---|---|---|---|---|---|---|---|')
        c = S['drift10']
        for arm in ('R3', 'I1_shuffled', 'I1_time', 'I1_shuffled_track', 'Tr3'):
            if arm not in c:
                continue
            sds = sorted(c[arm])
            v_ = lambda lab, k: np.mean([c[arm][s_][0][lab][k] for s_ in sds])
            P(f"| {SHORT.get(arm, arm)} | {v_('T','acc'):.3f} | {v_('C','acc'):.3f} | {v_('T','found'):.2f} | {v_('C','found'):.2f} | {v_('T','n_false'):.1f} | {v_('C','n_false'):.1f} | " + ' '.join(f"{c[arm][s_][0]['C']['acc']:.3f}" for s_ in sds) + ' |')

# ---- post hoc
post = load('posthoc')
if post:
    P('\n### Post hoc (not preregistered): I dictionary rebuilt from its final extraction; I given 2 passes (seeds 1-6)\n')
    P('| variant | measure | R1 (1) | R2 (2) | R3 (3) | I(1) end (1) | I(1) rebuilt (2) | I(2) at 1 pass | I(2) end (2) | I(2) rebuilt (3) | I(1) rebuilt − R2 | I(2) rebuilt − R3 | I(2) end − R2 |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    H = {}
    for d in post:
        v, s_ = d['job']['variant'], d['job']['seed']
        for a in d['arms']:
            H[(v, s_, a['name'])] = a['records']

    def tag(recs, t):
        return [r for r in recs if r['tag'] == t][-1]

    def near(recs, x, key):
        c = [(float(r['tag'][2:]), r[key]) for r in recs if r['tag'].startswith('it')]
        return min(c, key=lambda q: abs(q[0] - x))[1]

    sd = sorted({k[1] for k in H})
    for v in ('stationary', 'nonstat'):
        for key in ('te_med', 'te_mean'):
            r1 = np.array([get(v, s_, 'R1', None, key) for s_ in sd]); r2 = np.array([get(v, s_, 'R2', None, key) for s_ in sd]); r3 = np.array([get(v, s_, 'R3', None, key) for s_ in sd])
            i1e = np.array([tag(H[(v, s_, 'I1_shuffled')], 'final')[key] for s_ in sd])
            i1r = np.array([tag(H[(v, s_, 'I1_shuffled')], 'refit')[key] for s_ in sd])
            i2m = np.array([near(H[(v, s_, 'I2_shuffled')], 1.0, key) for s_ in sd])
            i2e = np.array([tag(H[(v, s_, 'I2_shuffled')], 'final')[key] for s_ in sd])
            i2r = np.array([tag(H[(v, s_, 'I2_shuffled')], 'refit')[key] for s_ in sd])
            P(f'| {v} | {key} | {r1.mean():.3f} | {r2.mean():.3f} | {r3.mean():.3f} | {i1e.mean():.3f} | {i1r.mean():.3f} | {i2m.mean():.3f} | {i2e.mean():.3f} | {i2r.mean():.3f} | {fmt(boot(i1r - r2))} | {fmt(boot(i2r - r3))} | {fmt(boot(i2e - r2))} |')
    P('\nAccuracy and found rate of I(2):\n')
    P('| variant | R3 C acc | I(1) T | I(1) C | I(2) T | I(2) C | I(2) C − R3 C | I(2) T − R3 C | I(2) found C − R3 | I(2) sparse found C / R3 |')
    P('|---|---|---|---|---|---|---|---|---|---|')
    for v in ('stationary', 'nonstat'):
        q = lambda lab, k: np.array([tag(H[(v, s_, 'I2_shuffled')], 'final')[lab][k] for s_ in sd])
        r3a = np.array([get(v, s_, 'R3', 'C', 'acc') for s_ in sd]); r3f = np.array([get(v, s_, 'R3', 'C', 'found') for s_ in sd])
        i1 = lambda lab: np.mean([tag(H[(v, s_, 'I1_shuffled')], 'final')[lab]['acc'] for s_ in sd])
        P(f"| {v} | {r3a.mean():.3f} | {i1('T'):.3f} | {i1('C'):.3f} | {q('T','acc').mean():.3f} | {q('C','acc').mean():.3f} | {fmt(boot(q('C','acc') - r3a))} | {fmt(boot(q('T','acc') - r3a))} | {fmt(boot(q('C','found') - r3f))} | {np.nanmean(q('C','found_lt05')):.2f} / {np.nanmean([get(v, s_, 'R3', 'C', 'found_lt05') for s_ in sd]):.2f} |")

if scale and 'T360' in S:
    P('\n### Is the template-error deficit a spike-count effect? (post hoc, seeds 1-3)\n')
    P('I(1) on a 360-s recording, at 1/3 of a pass (the same data volume as a 120-s pass) and at the end, against R3 on 360 s; units with rate > 3 Hz are not separated here.\n')
    P('| seed | R3, 120 s | I(1), 120 s | R3, 360 s | I(1), 360 s, at 1/3 pass | I(1), 360 s, end |')
    P('|---|---|---|---|---|---|')
    rows_ = []
    for d in scale:
        if d['job']['id'].endswith('_T360'):
            s_ = d['job']['seed']
            rr = {a['name'].split('|')[1]: a['records'] for a in d['arms']}
            c = [(float(r['tag'][2:]), r['te_med']) for r in rr['I1_shuffled'] if r['tag'].startswith('it')]
            third = min(c, key=lambda q: abs(q[0] - 1 / 3))[1]
            end = [r for r in rr['I1_shuffled'] if r['tag'] == 'final'][-1]['te_med']
            r3 = [r for r in rr['R3'] if r['tag'] == 'round3'][-1]['te_med']
            rows_.append((s_, E[('stationary', s_, 'R3')]['te_med'], E[('stationary', s_, 'I1_shuffled')]['te_med'], r3, third, end))
    for r_ in sorted(rows_):
        P(f'| {r_[0]} | {r_[1]:.3f} | {r_[2]:.3f} | {r_[3]:.3f} | {r_[4]:.3f} | {r_[5]:.3f} |')

# ---- extension
ext = load('ext2')
if ext:
    P('\n### Post hoc: best-case I against R extended past its grid edge (seeds 3-8; settings chosen on seeds 1-2)\n')
    P('Ibest: n_birth 8, spike schedule capped at 200, 0.5-s batches (buffer, birth and triage intervals kept in seconds). Ibest2s: the same at 2-s batches. Rx: R3 with the extension chosen in `ext_R.json`.\n')
    X = {}
    for d in ext:
        v, s_ = d['job']['variant'], d['job']['seed']
        for a_ in d['arms']:
            r = [x for x in a_['records'] if x['tag'] in ('final', 'round3')][-1]
            X[(v, s_, a_['name'])] = r
    sd = sorted({k[1] for k in X})
    P('| variant | output | acc | found | false | merged | sparse found | pair | te med | Δ acc vs R3 C (tuned) | Δ acc vs Rx C | Δ te vs R3 |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for v in ('stationary', 'nonstat'):
        base = np.array([E[(v, s_, 'R3')]['C']['acc'] for s_ in sd])
        bte = np.array([E[(v, s_, 'R3')]['te_med'] for s_ in sd])
        rx = np.array([X[(v, s_, 'Rx')]['C']['acc'] for s_ in sd])
        rows__ = [('R3 tuned', 'C', [E[(v, s_, 'R3')] for s_ in sd]), ('Rx', 'C', [X[(v, s_, 'Rx')] for s_ in sd])]
        for nm in ('Ibest', 'Ibest2s'):
            for lab in ('T', 'C'):
                rows__.append((nm, lab, [X[(v, s_, nm)] for s_ in sd]))
        for nm, lab, rr in rows__:
            q_ = lambda k: np.nanmean([r[lab][k] for r in rr])
            acc = np.array([r[lab]['acc'] for r in rr]); te_ = np.array([r['te_med'] for r in rr])
            P(f"| {v} | {nm} {lab} | {q_('acc'):.3f} | {q_('found'):.3f} | {q_('n_false'):.1f} | {q_('merged'):.3f} | {q_('found_lt05'):.2f} | {q_('pair_res'):.2f} | {te_.mean():.3f} | {fmt(boot(acc - base))} | {fmt(boot(acc - rx))} | {fmt(boot(te_ - bte))} |")

# ---- cap lifted
e3 = load('ext3')
if e3:
    P('\n### Post hoc: crowded conditions with the cluster cap lifted (K_init 32 at 192 units, 16 at 96; seeds 1-3)\n')
    P('| condition | output | acc | found | false | K | te med | per seed acc |')
    P('|---|---|---|---|---|---|---|---|')
    Z = {}
    for d in e3:
        for a_ in d['arms']:
            c_, nm = a_['name'].split('|')
            r = [x for x in a_['records'] if x['tag'] in ('final', 'round3')][-1]
            Z.setdefault((c_, nm), {})[d['job']['seed']] = r
    for c_ in ('u96', 'u192'):
        for nm, lab in (('O', 'T'), ('O', 'C'), ('R3k', 'C'), ('R3k', 'T'), ('Ik', 'T'), ('Ik', 'C')):
            rr = Z[(c_, nm)]; sd = sorted(rr)
            q_ = lambda k: np.mean([rr[s_][lab][k] for s_ in sd])
            P(f"| {c_} | {nm} {lab} | {q_('acc'):.3f} | {q_('found'):.2f} | {q_('n_false'):.1f} | {np.mean([rr[s_]['K'] for s_ in sd]):.0f} | {np.mean([rr[s_]['te_med'] for s_ in sd]):.3f} | " + ' '.join(f"{rr[s_][lab]['acc']:.3f}" for s_ in sd) + ' |')

txt = '\n'.join(out)
open(os.path.join(OUT, 'results.md'), 'w', encoding='utf8').write(txt)
sys.stdout.reconfigure(encoding='utf8')
print(txt)
