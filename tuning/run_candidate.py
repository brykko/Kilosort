"""Run one evaluation of the settings search (spec §3, §9, §11): sort, clean, evaluate, compare with
default, log.

    python run_candidate.py freeze
        Hash the harness (tuning/*.py, the spec, the round plans, config.json), the KS4 sources
        (kilosort/*.py, kilosort/cuda/*.cu and the built kernel library) and the frozen inputs
        (theta_ref, raw-artefact mask, hybrid truth; hybrid excerpt by size and modification time),
        record them with the git commit in frozen.json and as the first record of the search log.
        Refuses if tuning/ or kilosort/ has uncommitted changes or the log already exists.

    python run_candidate.py run --name NAME --settings '{"Th_learned": 7}' --seed 1 --rationale "..."
        Verifies every hash against frozen.json (refuses to score on any mismatch), writes a 'start'
        record to the log, sorts the hybrid excerpt in a separate process (run_sort.py), evaluates and
        compares in a separate process, then writes a 'result' record. Prints only what the optimising
        agent may see (§3): S, dS, S_track, C, guard values and pass/fail, unit counts, cleaning
        statistics, runtime. Per-unit scores stay in the run folder's evaluation.json, which the
        optimising agent does not read.

    python run_candidate.py refreeze --reason "spec v0.8: ..."
        After a spec or harness change during the search: same checks and hashes as freeze, appended
        to the existing log as a new 'freeze' record with the reason. Later runs verify against it.

Options: --phase (search, replicate, ablation, test), --existing-sort FOLDER (re-score an existing sort
instead of sorting; used for tests and for re-scoring after a spec change), --unfrozen (skip the hash
check; test phase only, logged to the separate test log so the search log stays clean).
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
BASE = Path(CFG['excerpt_dir'])
PY = sys.executable


def path(key):
    return BASE / CFG[key]


def sha256(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def current_hashes():
    harness = sorted([*HERE.glob('*.py'), HERE / 'evaluation_spec.md', *HERE.glob('search_round*.md'),
                      HERE / 'config.json'])
    ks = HERE.parent / 'kilosort'
    kilosort = sorted([*ks.glob('*.py'), *(ks / 'cuda').glob('*.cu'), *(ks / 'cuda').glob('*.dll'),
                       *(ks / 'cuda').glob('*.so')])
    data = path('data').stat()
    return {
        'harness': {f.name: sha256(f) for f in harness},
        'kilosort': {f.relative_to(ks).as_posix(): sha256(f) for f in kilosort},
        'reference': sha256(path('reference')),
        'raw_artefacts': sha256(path('raw_artefacts')),
        'hybrid_truth': sha256(path('hybrid_truth')),
        'hybrid_excerpt': {'size': data.st_size, 'mtime': data.st_mtime},
    }


LOG = 'log'          # switched to 'test_log' for --unfrozen runs


def log(record):
    record = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'), **record}
    with open(path(LOG), 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def read_log():
    p = path(LOG)
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []


def git(*a):
    return subprocess.run(['git', '-C', str(HERE.parent), *a], capture_output=True, text=True, check=True).stdout


def freeze(reason=None):
    """reason=None: first freeze, no log yet. Otherwise a re-freeze appended to the existing log."""
    if reason is None and path('log').exists():
        raise SystemExit(f"{path('log')} already exists; the search log is append-only (use refreeze)")
    if reason is not None and not path('log').exists():
        raise SystemExit('no search log yet; use freeze')
    if git('status', '--porcelain', '--', 'tuning', 'kilosort').strip():
        raise SystemExit('tuning/ or kilosort/ has uncommitted changes; commit before freezing')
    frozen = {'commit': git('rev-parse', 'HEAD').strip(), 'hashes': current_hashes()}
    if reason is not None:
        frozen['reason'] = reason
    path('frozen').write_text(json.dumps(frozen, indent=2))
    log({'type': 'freeze', **frozen})
    print(f"frozen at {frozen['commit'][:7]}; {len(frozen['hashes']['harness'])} harness and "
          f"{len(frozen['hashes']['kilosort'])} KS4 files hashed")


def verify():
    if not path('frozen').exists():
        raise SystemExit('harness not frozen; run "freeze" first (or --unfrozen with --phase test)')
    frozen, now = json.loads(path('frozen').read_text())['hashes'], current_hashes()
    bad = [k for k in now if now[k] != frozen[k]]
    if bad:
        detail = [f for k in ('harness', 'kilosort') if k in bad
                  for f in set(now[k]) | set(frozen[k]) if now[k].get(f) != frozen[k].get(f)]
        raise SystemExit(f'hash mismatch, refusing to score: {bad} {detail}')
    return now


def visible_summary(ev, cmp):
    """What the optimising agent may see (§3)."""
    return {
        'S': ev['S'], 'dS': cmp['dS'], 'S_default': cmp['S_default'], 'S_track': ev['S_track'],
        'dS_track': cmp['dS_track'],
        'guards': [{k: g[k] for k in ('guard', 'candidate', 'default', 'tau', 'status')} for g in cmp['guards']],
        'all_guards_pass': all(g['status'] == 'pass' for g in cmp['guards']),
        'n_units': ev['n_units'], 'n_eligible': ev['n_eligible'], 'n_noise_units': ev['n_noise_units'],
        'fraction_spikes_removed': ev['fraction_spikes_removed'], 'n_flagged_pairs': ev['n_flagged_pairs'],
        'multiunit_events': ev['multiunit_events'],
        'hybrid': {k: ev['hybrid'][k] for k in ('median_agreement', 'n_recovered', 'n_inserted')}
                  | {'n_merged_pairs': len(ev['hybrid']['merged_pairs'])},
        'matched_units': {'n_matched': cmp['matched_units']['n_matched'],
                          'unmatched_candidate': cmp['matched_units']['unmatched_candidate']['n'],
                          'unmatched_default': cmp['matched_units']['unmatched_default']['n']},
    }


def run(args):
    settings = json.loads(args.settings)
    fixed = [k for k in settings if k in CFG['fixed_settings']]
    if fixed:
        raise SystemExit(f'fixed settings may not vary (spec §4): {fixed}')
    if not 1 <= args.seed <= CFG['max_seed']:
        raise SystemExit(f"seed must be 1-{CFG['max_seed']} (seeds >= 1000 are reserved for reference sorts)")
    if args.unfrozen and args.phase != 'test':
        raise SystemExit('--unfrozen is only allowed with --phase test')
    global LOG
    if args.unfrozen:
        LOG = 'test_log'
    hashes = current_hashes() if args.unfrozen else verify()

    records = read_log()
    n = sum(r.get('type') == 'start' for r in records) + 1
    counted = sum(r.get('type') == 'start' and r['phase'] != 'test' and not r.get('existing_sort') for r in records)
    if args.phase == 'search' and counted >= CFG['budget'] - CFG['reserved_for_selection']:
        raise SystemExit('search budget used up; the remaining runs are reserved for replication and ablation')
    run_id = f'{n:03d}_{args.name}_s{args.seed}'
    folder = path('search_runs') / ('test_' + run_id if args.unfrozen else run_id)
    log({'type': 'start', 'run_id': run_id, 'phase': args.phase, 'seed': args.seed, 'settings': settings,
         'rationale': args.rationale, 'existing_sort': args.existing_sort, 'unfrozen': args.unfrozen,
         'reference_sha256': hashes['reference'], 'harness_sha256': hashes['harness']})

    t_start = time.time()
    sort = Path(args.existing_sort) if args.existing_sort else folder / 'sort'
    folder.mkdir(parents=True, exist_ok=False)
    if not args.existing_sort:
        subprocess.run([PY, str(HERE / 'run_sort.py'), '--data', str(path('data')), '--probe', str(path('probe')),
                        '--n-chan-bin', str(CFG['n_chan_bin']), '--out', str(sort), '--seed', str(args.seed),
                        '--settings', json.dumps(settings)],
                       check=True, stdout=open(folder / 'sort.log', 'w'), stderr=subprocess.STDOUT)
    t_sort = time.time() - t_start
    common = ['--raw-artefacts', str(path('raw_artefacts'))]
    with open(folder / 'harness.log', 'w') as f:
        subprocess.run([PY, str(HERE / 'evaluate.py'), '--sort', str(sort), '--reference', str(path('reference')),
                        '--t0', str(CFG['t0']), '--duration', str(CFG['duration']),
                        '--hybrid-truth', str(path('hybrid_truth')), *common], check=True, stdout=f, stderr=subprocess.STDOUT)
        subprocess.run([PY, str(HERE / 'compare.py'), '--candidate', str(sort), '--out', str(folder / 'comparison.json'),
                        '--default', *[str(BASE / d) for d in CFG['default_replicates']], *common],
                       check=True, stdout=f, stderr=subprocess.STDOUT)
    ev = json.loads((sort / 'evaluation.json').read_text())
    cmp = json.loads((folder / 'comparison.json').read_text())
    summary = visible_summary(ev, cmp)
    summary.update(run_id=run_id, sort_min=t_sort / 60, total_min=(time.time() - t_start) / 60)
    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    log({'type': 'result', 'run_id': run_id, **summary})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('freeze')
    rf = sub.add_parser('refreeze')
    rf.add_argument('--reason', required=True)
    r = sub.add_parser('run')
    r.add_argument('--name', required=True)
    r.add_argument('--settings', default='{}')
    r.add_argument('--seed', type=int, default=1)
    r.add_argument('--rationale', required=True)
    r.add_argument('--phase', default='search', choices=['search', 'replicate', 'ablation', 'test'])
    r.add_argument('--existing-sort')
    r.add_argument('--unfrozen', action='store_true')
    args = p.parse_args()
    if args.cmd == 'run':
        run(args)
    else:
        freeze(args.reason if args.cmd == 'refreeze' else None)
