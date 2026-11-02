"""Run a queue of search evaluations through run_candidate.py, one sort at a time.

The next run's sort starts as soon as the previous run's sort has finished (its harness evaluation
runs on the CPU meanwhile). Prints one line per run from its summary.json: only what the optimising
agent may see (spec §3).

    python run_queue.py queue.json      # [[name, settings, rationale, phase, seed], ...]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
RUNS = Path(CFG['excerpt_dir']) / CFG['search_runs']


def line(s):
    fails = [g['guard'] for g in s['guards'] if g['status'] != 'pass']
    return (f"{s['run_id']}: dS {s['dS']:+.1f} dS_track {s['dS_track']:+.1f} eligible {s['n_eligible']} "
            f"units {s['n_units']} hybrid {s['hybrid']['n_recovered']}/{s['hybrid']['n_inserted']} "
            f"sort {s['sort_min']:.1f} min | {'all guards pass' if not fails else 'FAIL: ' + '; '.join(fails)}")


def report(proc, name):
    out, err = proc.communicate()
    if proc.returncode:
        print(f'[queue] {name} FAILED (exit {proc.returncode}): {err.strip().splitlines()[-1:]}', flush=True)
        return False
    run_id = json.loads(out[out.index('{'):])['run_id']
    print('[result] ' + line(json.loads((RUNS / run_id / 'summary.json').read_text())), flush=True)
    return True


if __name__ == '__main__':
    queue = json.loads(Path(sys.argv[1]).read_text())
    pending = None                                     # (proc, name) of a run still in its harness stage
    for name, settings, why, phase, seed in queue:
        print(f'[queue] starting {name} seed {seed}', flush=True)
        before = {p.name for p in RUNS.iterdir()}
        proc = subprocess.Popen([sys.executable, str(HERE / 'run_candidate.py'), 'run', '--name', name,
                                 '--settings', json.dumps(settings), '--seed', str(seed), '--rationale', why,
                                 '--phase', phase], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        folder = None
        while proc.poll() is None:                     # wait until this run's sort is done
            if folder is None:
                new = [p for p in RUNS.iterdir() if p.name not in before and p.name.endswith(f'_{name}_s{seed}')]
                folder = new[0] if new else None
            if folder is not None and (folder / 'harness.log').exists():
                break
            time.sleep(5)
        if pending and not report(*pending):
            sys.exit(1)
        pending = (proc, name)
    if pending:
        report(*pending)
    print('[queue] done', flush=True)
