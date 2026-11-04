"""Quick synthetic check (report writing): does the sign of MSR's response to Poisson contamination
depend on burstiness? Uses the repo MSR implementation (tuning/exploratory/msr/msr.py) and the theory
note's generators (tuning/exploratory/msr/theory/msr.py). 30% added Poisson spikes, paired, 10 seeds."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

MSRDIR = Path('C:/code/github/Kilosort/tuning/exploratory/msr')
sys.path.insert(0, str(MSRDIR))
import msr as M  # repo implementation  # noqa: E402
spec = importlib.util.spec_from_file_location('tmsr', MSRDIR / 'theory' / 'msr.py')
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

dur = 1200.0


def bursts(rng, onset_rate, isi=0.004, kmin=2, kmax=10):
    on = tm.poisson(onset_rate, dur, rng)
    k = rng.integers(kmin, kmax + 1, on.size)
    t = np.concatenate([o + isi * np.arange(n) for o, n in zip(on, k)])
    return np.sort(t[t < dur])


def modulated_bursts(rng):
    # burst onsets follow the multiscale rate; bursts of 2-6 spikes 5 ms apart
    on = tm.inhom(tm.multiscale_rate(dur, 1.4, rng), dur, rng)
    k = rng.integers(2, 7, on.size)
    t = np.concatenate([o + 0.005 * np.arange(n) for o, n in zip(on, k)])
    return np.sort(t[t < dur])


def mixed(rng):
    # modulated unit where ~45% of spikes are in bursts: OU unit plus burst companions
    t = tm.refractory(tm.inhom(tm.multiscale_rate(dur, 3, rng), dur, rng), 0.002)
    sel = t[rng.random(t.size) < 0.35]
    comp = np.concatenate([sel + 0.004, sel + 0.008])
    return np.sort(np.concatenate([t, comp[rng.random(comp.size) < 0.6]]))


gens = {
    'OU multiscale 5 Hz + 2 ms dead time (theory U)': lambda r: tm.refractory(tm.inhom(tm.multiscale_rate(dur, 5, r), dur, r), 0.002),
    'Poisson 5 Hz': lambda r: tm.poisson(5, dur, r),
    'gamma shape 0.3 renewal 5 Hz': lambda r: tm.gamma_renewal(5, 0.3, dur, r),
    'Poisson-onset bursts (2-10 spikes, 4 ms)': lambda r: bursts(r, 0.83),
    'OU-modulated bursts (2-6 spikes, 5 ms)': modulated_bursts,
    'OU unit with burst companions (~half ISIs <20 ms)': mixed,
}
for name, g in gens.items():
    d, m0s, fr = [], [], []
    for s in range(10):
        r = np.random.default_rng(100 + s)
        t = g(r)
        m0 = M.msr(t, dur)
        tc = np.sort(np.concatenate([t, r.uniform(0, dur, int(0.3 * t.size))]))
        d.append(M.msr(tc, dur) - m0)
        m0s.append(m0)
        fr.append(np.mean(np.diff(t) < 0.02))
    d = np.array(d)
    print(f'{name:52s} MSR {np.mean(m0s):.4f}  ISI<20ms {np.mean(fr):.2f}  dMSR(+30%) {d.mean():+.5f} '
          f'(SE {d.std(ddof=1) / np.sqrt(d.size):.5f})')
