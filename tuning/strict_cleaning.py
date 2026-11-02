"""Duplicate robustness at selection (spec §9.2): re-score sorts with stricter duplicate cleaning.

Cleaning (§6.3) removes duplicates only where a pair's excess of coincident spikes is significant
and >= MIN_EXCESS (3%) of the smaller unit's spikes; a sort with many duplicates may keep some
below that line. Here each sort is cleaned and scored again with MIN_EXCESS = 1%, in a shadow folder
of hard links (the sort's own files and its cleaning and evaluation stay untouched). The winner must
still beat default, re-scored the same way, by >= delta.

    python strict_cleaning.py --sorts <sort> ... --out <json>
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import cleaning
from evaluate import evaluate

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text())
BASE = Path(CFG['excerpt_dir'])
STRICT_EXCESS = 0.01


def shadow(sort):
    """Folder of hard links to the sort's files, without its cleaning and evaluation results."""
    sort = Path(sort)
    out = BASE / 'runs' / 'strict_cleaning' / (sort.parent.name if sort.name == 'sort' else sort.name)
    out.mkdir(parents=True, exist_ok=True)
    for f in sort.iterdir():
        if f.is_file() and not f.name.startswith(('cleaning', 'evaluation')) and not (out / f.name).exists():
            os.link(f, out / f.name)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sorts', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    cleaning.MIN_EXCESS = STRICT_EXCESS
    res = {}
    for s in args.sorts:
        d = shadow(s)
        ev = evaluate(d, BASE / CFG['reference'], BASE / CFG['raw_artefacts'], CFG['t0'], CFG['duration'],
                      BASE / CFG['hybrid_truth'])
        res[str(s)] = {'S_strict': ev['S'], 'n_eligible': ev['n_eligible']}
        print(f"{s}: S {ev['S']:.1f} with stricter cleaning, {ev['n_eligible']} eligible units", flush=True)
    Path(args.out).write_text(json.dumps({'min_excess': STRICT_EXCESS, 'sorts': res}, indent=2))
