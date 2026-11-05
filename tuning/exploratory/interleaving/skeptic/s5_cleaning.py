"""What the rounds' extra spikes are: extracted, removed by KS4's own duplicate step, removed by harness cleaning, and
kept, by round; plus same-unit short-lag events per 1000 spikes (0.3-0.53 ms, the echo window of
duplicates_audit.md). All seeds, both excerpts, H8 and default, augment."""
from common import *  # noqa


def row(d):
    o = ops_of(d)
    rounds = o.get('template_rounds')
    c = json.loads((d / 'cleaning.json').read_text())
    n = c['n_spikes']
    t = np.load(d / 'spike_times.npy').ravel().astype(np.int64)
    u = np.load(d / 'spike_clusters.npy').ravel().astype(np.int64)
    ord_ = np.lexsort((t, u))
    dt = np.diff(t[ord_])
    same = u[ord_][1:] == u[ord_][:-1]
    echo = ((dt >= 9) & (dt <= 16) & same).sum()
    near = ((dt >= 17) & (dt <= 45) & same).sum()     # 0.57-1.5 ms, same unit
    return dict(extracted=rounds[-1]['n_spikes'] if rounds else None, kept_ks4=n,
                cleaned=c['fraction_spikes_removed'], after_clean=n * (1 - c['fraction_spikes_removed']),
                dup_pairs=c['duplicates'].get('n_flagged', c['duplicates'].get('flagged_pairs', None)) if isinstance(c['duplicates'], dict) else None,
                dup_spikes=c['duplicate_spikes_removed'], event_spikes=c['events_spikes_removed'],
                echo_per_1000=1000 * echo / n, refr_0p57_1p5_per_1000=1000 * near / n)


res = {}
for ex in EXS:
    for setting, rounds in (('H8', (2, 3, 4)), ('default', (2, 3)), ('aug', (3,))):
        for k in rounds:
            rows = []
            for s in SEEDS:
                d = folder(ex, setting, k, s)
                if not (d / 'cleaning.json').exists():
                    continue
                rows.append(row(d))
            res[f'{ex}_{setting}_r{k}'] = rows
            m = {key: np.mean([r[key] for r in rows]) for key in rows[0] if rows[0][key] is not None}
            print(ex, setting, k, {a: round(float(b), 4) for a, b in m.items()}, flush=True)
(OUT / 's5_cleaning.json').write_text(json.dumps(res, indent=1, default=float))
