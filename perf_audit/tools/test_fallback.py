"""Check the perf-optimizations branch against KS4's original template_match and
run_matching (loaded from the main checkout), on the same batches:
  GPU, fused kernels | GPU, PyTorch fallback | CPU, PyTorch fallback (one batch).

Usage: python test_fallback.py RUN_DIR (a profile_run --snapshot folder)
"""
import importlib.util
import sys
from pathlib import Path

BRANCH = r'C:\code\github\Kilosort-perf'
MAIN = Path(r'C:\code\github\Kilosort\kilosort')
sys.path.insert(0, BRANCH)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import kilosort  # noqa: E402
from kilosort import io, spikedetect, template_matching, cuda_kernels  # noqa: E402

assert Path(kilosort.__file__).parent == Path(BRANCH, 'kilosort'), kilosort.__file__


def load_original(name):
    spec = importlib.util.spec_from_file_location(f'orig_{name}', MAIN / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


orig_sd, orig_tm = load_original('spikedetect'), load_original('template_matching')
run = Path(sys.argv[1])


def setup(dev):
    ops = io.load_ops(run / 'ops.npy', device=dev)
    bfile = io.bfile_from_ops(ops, device=dev)
    U = torch.load(run / 'snapshot' / 'U_first.pt').to(dev)
    template_matching.prepare_extract(ops['xc'], ops['yc'], U, ops['settings']['nearest_chans'],
                                      ops['settings']['position_limit'], device=dev)
    return ops, bfile, U, template_matching.prepare_matching(ops, U)


def same(a, b):
    return all(x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b))


def compare(dev, batches, modes):
    ops, bfile, U, ctc = setup(dev)
    ok = {m: [0, 0] for m in modes}
    for ib in batches:
        X = bfile.padded_batch_to_torch(int(ib), ops)
        ref_d = orig_sd.template_match(X, ops, ops['iC'], ops['iC2'], ops['weigh'], device=dev)
        ref_m = orig_tm.run_matching(ops, X, U, ctc, device=dev)
        for m in modes:
            cuda_kernels._loaded, cuda_kernels._lib = (False, None) if m == 'fused' else (True, None)
            d = spikedetect.template_match(X, ops, ops['iC'], ops['iC2'], ops['weigh'], device=dev)
            r = template_matching.run_matching(ops, X, U, ctc, device=dev)
            ok[m][0] += same(ref_d, d)
            ok[m][1] += same(ref_m, r)
    for m in modes:
        print(f'{dev.type}/{m:8s}: template_match identical {ok[m][0]}/{len(batches)}, '
              f'run_matching identical {ok[m][1]}/{len(batches)}')


cuda_kernels._load()
assert cuda_kernels._lib is not None, 'kernel library not built'
compare(torch.device('cuda'), np.linspace(0, 599, 20).astype(int), ['fused', 'fallback'])
compare(torch.device('cpu'), [300], ['fallback'])
