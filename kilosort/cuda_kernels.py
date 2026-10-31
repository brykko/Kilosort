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
KERNELS_VERSION = 2   # must match ks_version() in kilosort_kernels.cu
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
    # A library built from older sources may export the same names with other
    # signatures, so it must report the version these bindings were written for.
    version = getattr(lib, 'ks_version', None)
    version = version() if version is not None else None
    if version != KERNELS_VERSION:
        logger.warning(f'{LIBRARY_PATH.name} is out of date (version {version}, need '
                       f'{KERNELS_VERSION}); rebuild it with kilosort/cuda/build.bat or '
                       f'build.sh. Using PyTorch code.')
        return
    vp, ci = ctypes.c_void_p, ctypes.c_int
    lib.ks_template_match.argtypes = [vp]*7 + [ci]*6 + [ctypes.c_float, ci]
    lib.ks_matching_peaks.argtypes = [vp] + [ci]*2 + [ctypes.c_float] + [vp]*2
    lib.ks_matching_record.argtypes = [vp]*5 + [ci]*3 + [vp]*4
    lib.ks_matching_xres.argtypes = [vp]*4 + [ci]*5
    lib.ks_matching_xres_fast.argtypes = [vp]*6 + [ci]*6
    lib.ks_matching_bupdate.argtypes = [vp]*5 + [ci]*5
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


# run_matching, one peel step. Group arguments (iX, iY, amp of one interleaved group) are
# strided views; the kernels take their data pointers and element strides.

def matching_peaks(Cfmax, nt, thr, flags, out):
    """out[0] = number of peaks, out[1:] = their columns ascending. Returns False if
    nt is too large for the kernel (nothing computed)."""
    err = _lib.ks_matching_peaks(Cfmax.data_ptr(), Cfmax.shape[0], int(nt), float(thr),
                                 flags.data_ptr(), out.data_ptr())
    return _check(err, 'ks_matching_peaks')


def matching_record(B, nm, imax, Cfmax, iX, nt, st, amps, th_amps, iY):
    """Spike times, units, amplitudes and threshold amplitudes of the peaks iX."""
    err = _lib.ks_matching_record(
        B.data_ptr(), nm.data_ptr(), imax.data_ptr(), Cfmax.data_ptr(), iX.data_ptr(),
        len(iX), B.shape[1], int(nt), st.data_ptr(), amps.data_ptr(), th_amps.data_ptr(),
        iY.data_ptr()
        )
    return _check(err, 'ks_matching_record')


def matching_xres(Xres, E, iX, amp):
    """Xres[:, iX + tiwave] -= amp * E, with E (channels, spikes, nt) contiguous."""
    assert iX.stride(0) == amp.stride(0)
    err = _lib.ks_matching_xres(
        Xres.data_ptr(), E.data_ptr(), iX.data_ptr(), amp.data_ptr(), len(iX),
        iX.stride(0), Xres.shape[0], Xres.shape[1], E.shape[2]
        )
    return _check(err, 'ks_matching_xres')


def matching_xres_fast(Xres, U, W, iX, iY, amp):
    """As matching_xres, computing E = einsum('ijk, jl -> kil', U[iY], W) in the kernel
    (can differ from cuBLAS at float rounding)."""
    assert iX.stride(0) == iY.stride(0) == amp.stride(0)
    err = _lib.ks_matching_xres_fast(
        Xres.data_ptr(), U.data_ptr(), W.data_ptr(), iX.data_ptr(), iY.data_ptr(),
        amp.data_ptr(), len(iX), iX.stride(0), Xres.shape[0], Xres.shape[1], W.shape[1],
        W.shape[0]
        )
    return _check(err, 'ks_matching_xres_fast')


def matching_bupdate(B, ctc, iX, iY, amp, nt):
    """B[:, iX + trange] -= amp * ctc[:, iY, :] for spikes whose windows don't overlap."""
    assert iX.stride(0) == iY.stride(0) == amp.stride(0)
    err = _lib.ks_matching_bupdate(
        B.data_ptr(), ctc.data_ptr(), iX.data_ptr(), iY.data_ptr(), amp.data_ptr(),
        len(iX), iX.stride(0), B.shape[0], B.shape[1], int(nt)
        )
    return _check(err, 'ks_matching_bupdate')


def matching_colmax(B, nm, iX, nt, Cfmax, imax):
    """Max and argmax over units of relu(B)**2 / nm, on columns within nt of iX."""
    err = _lib.ks_matching_colmax(
        B.data_ptr(), nm.data_ptr(), iX.data_ptr(), len(iX), B.shape[0], B.shape[1],
        int(nt), Cfmax.data_ptr(), imax.data_ptr()
        )
    return _check(err, 'ks_matching_colmax')
