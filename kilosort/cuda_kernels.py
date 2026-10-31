"""Optional fused CUDA kernels, loaded with ctypes from a library built by
`kilosort/cuda/build.bat` (Windows) or `kilosort/cuda/build.sh` (Linux).

If the library hasn't been built, or computation is on the CPU, the callers in
`spikedetect` and `template_matching` use their PyTorch code instead. The kernels
give the same results as that code (bit for bit with default settings), only faster.

"""
import ctypes
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_name = 'kilosort_kernels.dll' if sys.platform == 'win32' else 'libkilosort_kernels.so'
LIBRARY_PATH = Path(__file__).parent / 'cuda' / _name
_lib = None
_loaded = False


def _load():
    global _lib, _loaded
    _loaded = True
    if not LIBRARY_PATH.exists():
        logger.info(f'Fused CUDA kernels not built ({LIBRARY_PATH.name}), using PyTorch code.')
        return
    try:
        lib = ctypes.CDLL(str(LIBRARY_PATH))
    except OSError as e:
        logger.warning(f'Could not load {LIBRARY_PATH}: {e}. Using PyTorch code.')
        return
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.ks_template_match.argtypes = [vp]*7 + [ci]*6 + [ctypes.c_float, ci]
    lib.ks_matching_bupdate.argtypes = [vp]*5 + [ci]*4
    lib.ks_matching_colmax.argtypes = [vp]*3 + [ci]*4 + [vp]*2
    _lib = lib
    logger.info(f'Using fused CUDA kernels from {LIBRARY_PATH}')


def available(device):
    """True if the fused kernels can be used for tensors on `device`."""
    if torch.device(device).type != 'cuda':
        return False
    if not _loaded:
        _load()
    return _lib is not None


def _check(err, name):
    if err > 0:
        raise RuntimeError(f'{name} failed with CUDA error {err}')
    return err == 0


def template_match(B, iC, weigh, As, imaxs, iC2, Amaxs, Th, nt0):
    """Fused inner loop of `spikedetect.template_match`. Returns False if the shapes
    aren't supported, in which case nothing was computed."""
    if (B.dtype != torch.float32 or weigh.dtype != torch.float32
            or iC.dtype != torch.int64 or iC2.dtype != torch.int64):
        return False
    err = _lib.ks_template_match(
        B.data_ptr(), iC.data_ptr(), weigh.data_ptr(), As.data_ptr(), imaxs.data_ptr(),
        iC2.data_ptr(), Amaxs.data_ptr(), B.shape[1], B.shape[2], iC.shape[0], iC.shape[1],
        weigh.shape[0], iC2.shape[0], float(Th), int(nt0)
        )
    return _check(err, 'ks_template_match')


def matching_bupdate(B, ctc, iX, iY, amp, nt):
    """B[:, iX + trange] -= amp * ctc[:, iY, :] for spikes whose windows don't overlap."""
    err = _lib.ks_matching_bupdate(
        B.data_ptr(), ctc.data_ptr(), iX.data_ptr(), iY.data_ptr(), amp.data_ptr(),
        len(iX), B.shape[0], B.shape[1], int(nt)
        )
    return _check(err, 'ks_matching_bupdate')


def matching_colmax(B, nm, cols, nt, Cfmax, imax):
    """Max and argmax over units of relu(B)**2 / nm, on columns `cols` only."""
    err = _lib.ks_matching_colmax(
        B.data_ptr(), nm.data_ptr(), cols.data_ptr(), len(cols), B.shape[0], B.shape[1],
        int(nt), Cfmax.data_ptr(), imax.data_ptr()
        )
    return _check(err, 'ks_matching_colmax')
