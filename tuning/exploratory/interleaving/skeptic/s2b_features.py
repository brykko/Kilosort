"""How much of the data do the templates reach? In extract(), a spike's features are residual + amp * template on the
template's 10 channels, so for a spike with nothing else subtracted near it they equal the raw projection, whatever
the template is. Check on the same spike in two sorts: is the feature norm (amplitudes.npy) identical? It can only
be identical if the channel window, the time and the neighbours' subtraction all had no effect.

Then split label changes: of the spikes whose unit has a stable partner, how many left the partner, among spikes
with unchanged features (no feature-based reason; pure re-clustering) and among spikes with changed features.
Stable pair and same spike as in s2_pairs.py. Same rule on every comparison, including the cross-seed controls.
"""
from common import *  # noqa
from s2_pairs import pair, stable


def analyse(tag, da, db, ex):
    a, b = spikes(da), spikes(db)
    ia, ib = pair(a, b)
    pairs, na, nb = stable(a, b, ia, ib)
    keep = -np.ones(int(a['clu'].max()) + 1, np.int64)
    for k, v in pairs.items():
        keep[k] = v
    has = keep[a['clu'][ia]] >= 0
    x, y = ia[has], ib[has]
    same_unit = keep[a['clu'][x]] == b['clu'][y]
    dla = np.abs(np.log(b['amp'][y] / a['amp'][x]))
    same = dla < 1e-5
    same_t = a['t'][x] == b['t'][y]
    o = dict(tag=tag, ex=ex, paired=float(ia.size / a['t'].size), with_partner=float(has.sum() / a['t'].size),
             features_identical=float(same.mean()), same_time=float(same_t.mean()),
             features_within_1pct=float((dla < .01).mean()),
             left_partner=float((~same_unit).mean()), left_given_identical=float((~same_unit[same]).mean()),
             left_given_changed=float((~same_unit[~same]).mean()),
             share_of_leavers_identical=float((same & ~same_unit).sum() / max((~same_unit).sum(), 1)))
    print(json.dumps(o), flush=True)
    return o


if __name__ == '__main__':
    J = []
    for ex in EXS:
        for s in SEEDS:
            J += [(f'H8 r2->r3 s{s}', folder(ex, 'H8', 2, s), folder(ex, 'H8', 3, s), ex),
                  (f'H8 r3->r4 s{s}', folder(ex, 'H8', 3, s), folder(ex, 'H8', 4, s), ex),
                  (f'default r2->r3 s{s}', folder(ex, 'default', 2, s), folder(ex, 'default', 3, s), ex),
                  (f'H8 r2->aug3 s{s}', folder(ex, 'H8', 2, s), folder(ex, 'aug', 3, s), ex)]
        J += [('H8 seeds r2 s1-s2', folder(ex, 'H8', 2, 1), folder(ex, 'H8', 2, 2), ex),
              ('H8 seeds r2 s2-s3', folder(ex, 'H8', 2, 2), folder(ex, 'H8', 2, 3), ex)]
    i, n = int(sys.argv[1]), int(sys.argv[2])
    with open(OUT / f's2b_{i}.jsonl', 'a') as h:
        for j, q in enumerate(J):
            if j % n == i:
                h.write(json.dumps(analyse(*q)) + '\n')
                h.flush()
