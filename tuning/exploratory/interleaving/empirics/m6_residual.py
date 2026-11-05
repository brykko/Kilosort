"""Measurement 6: waveforms left unexplained after matching, on a few batches, CPU.

The extraction dictionary of the last round isn't saved, so matching is rerun on five evenly spaced batches with the
saved final units' templates (Wall, aligned with align_U; no first-pass merge) as dictionary, with run_matching's own
threshold and peel limit: this is close to, not the same as, one more extraction. For memory and time on the CPU only a
1000 um band around the median unit depth is treated (templates peaking within 150 um of it; events counted inside it). Then the universal pass's own detector
(spikedetect.template_match, same threshold) runs on

    X      the preprocessed batch (check: reproduces st0's count there);
    Xres   the batch after subtraction;
    null   Xres with the same random phases applied to every channel (keeps spectrum and spatial covariance, destroys
           waveforms in time), so the rate that Gaussian-like residual noise alone produces.

An event on Xres is 'near' if a found spike lies within 10 samples and 50 um (imperfect subtraction), else 'isolated'
(a candidate for an unmodelled waveform; it failed Th_learned on every template of the dictionary and passes
Th_universal on a generic shape). A time-shifted control gives the chance rate of 'near'. Sign: + is the sign of the
universal templates' trough (a spike), - an over-subtraction.

    python m6_residual.py [--ex kobe] [--rounds 2 4] [--batches 100 200 300 400 500]
"""
import argparse
import json
import os
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import OUT, nearest_event, sort_dir  # noqa: E402
from kilosort.io import BinaryFiltered, load_ops  # noqa: E402
from kilosort.spikedetect import nearest_chans, template_match, yweighted  # noqa: E402
import kilosort.template_matching as tm  # noqa: E402
from kilosort.template_matching import align_U, prepare_matching, run_matching  # noqa: E402


_stock_bupdate = tm._bupdate


def _chunked_bupdate(B, ctc, iX, iY, amp, trange, block=128):
    """tm._bupdate in blocks: on the CPU path its index arrays for a whole peel reach several GB (14 GB of private
    memory in the first run). Same arithmetic: spikes of one call don't overlap."""
    for i in range(0, iX.numel(), block):
        _stock_bupdate(B, ctc, iX[i:i + block], iY[i:i + block], amp[i:i + block], trange)


tm._bupdate = _chunked_bupdate


def bfile_of(ops):
    fn = ops['filename']
    fn = fn[0] if isinstance(fn, (list, tuple)) else fn
    return BinaryFiltered(fn, ops['n_chan_bin'], fs=ops['fs'], NT=ops['batch_size'], nt=ops['nt'], nt0min=ops['nt0min'],
                          chan_map=ops['probe']['chanMap'], hp_filter=ops['fwav'], whiten_mat=ops['Wrot'],
                          dshift=ops['dshift'], device=torch.device('cpu'), do_CAR=ops['do_CAR'],
                          artifact_threshold=ops['artifact_threshold'], invert_sign=ops['invert_sign'],
                          dtype=ops['data_dtype'], tmin=ops['tmin'], tmax=ops['tmax'], shift=ops['shift'], scale=ops['scale'])


def signed(X, ops, iC, iC2, weigh, yc):
    """template_match's events: times, depths, amplitudes, and each event's sign (template_match returns imax without it)."""
    dev = torch.device('cpu')
    xy, imax, amp, adist = template_match(X, ops, iC, iC2, weigh, device=dev)
    y = yweighted(yc, iC, adist, xy, device=dev)
    W = ops['wTEMP']
    k = imax % ops['settings']['n_templates']
    B = torch.nn.functional.conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=ops['nt'] // 2)
    s = torch.einsum('ij, ji -> i', weigh[imax // ops['settings']['n_templates'], :, xy[:, 0]],
                     B[iC[:, xy[:, 0]], k, xy[:, 1]]).sign()
    return xy[:, 1].numpy().astype(np.int64), y.numpy().astype(np.float32), amp.numpy(), s.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ex', default='kobe')
    p.add_argument('--rounds', type=int, nargs='+', default=[2, 4])
    p.add_argument('--batches', type=int, nargs='+', default=[100, 200, 300, 400, 500])
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device('cpu')
    rng = np.random.default_rng(0)
    st0 = np.load(OUT / f'st0_{a.ex}_s1.npy')
    out = {}
    for rnd in a.rounds:
        d = sort_dir(a.ex, 'H8', 1, rnd)
        ops = load_ops(d / 'ops.npy', device=dev)
        ops['Th_universal'] = ops['settings']['Th_universal']
        ops['Th_learned'] = ops['settings']['Th_learned']
        ops['max_peels'] = ops['settings']['max_peels']
        ops['nt'] = ops['settings']['nt']
        Wall = torch.from_numpy(np.load(d / 'Wall.npy')).float()
        Wall = Wall[(Wall ** 2).sum((1, 2)) > 0]
        U, _ = align_U(Wall, ops, device=dev)
        # a 1000 um band centred on the median unit depth, with a 150 um margin of templates on each side
        yU = np.asarray(ops['yc'].cpu() if isinstance(ops['yc'], torch.Tensor) else ops['yc'])[torch.argmax((U ** 2).sum(1), -1).numpy()]
        med = np.median(yU)
        lo, hi = med - 500, med + 500
        U = U[torch.from_numpy((yU >= lo - 150) & (yU <= hi + 150))]
        print('band', lo, hi, 'templates', U.shape[0], 'of', yU.size, flush=True)
        t0 = time.time()
        ctc = prepare_matching(ops, U)
        print('ctc', time.time() - t0, flush=True)
        iU = torch.argmax((U ** 2).sum(1), -1)
        ch_y = np.asarray(ops['yc'])[iU.numpy()]
        bf = bfile_of(ops)
        # the detector on the band's universal centres only (5x faster on the CPU; same events inside the band)
        ycup, xcup = np.asarray(ops['ycup'].cpu() if isinstance(ops['ycup'], torch.Tensor) else ops['ycup']), np.asarray(ops['xcup'].cpu() if isinstance(ops['xcup'], torch.Tensor) else ops['xcup'])
        kc = (ycup >= lo - 150) & (ycup <= hi + 150)
        kt = torch.from_numpy(kc)
        iC, weigh = ops['iC'][:, kt], ops['weigh'][:, :, kt]
        iC2, _ = nearest_chans(ycup[kc], ycup[kc], xcup[kc], xcup[kc], ops['settings']['nearest_templates'], device=dev)
        yc = np.asarray(ops['yc'].cpu() if isinstance(ops['yc'], torch.Tensor) else ops['yc'])
        rows = []
        for ib in a.batches:
            t0 = time.time()
            X = bf.padded_batch_to_torch(ib, ops)
            stt, amps, th_amps, Xres = run_matching(ops, X, U, ctc, device=dev)
            tm = time.time() - t0
            tf = stt[:, 0].numpy().astype(np.int64)
            yf = ch_y[stt[:, 1].numpy()].astype(np.float32)
            o = np.argsort(tf)
            tf, yf = tf[o], yf[o]
            found_band = int(((yf >= lo) & (yf <= hi)).sum())
            ev = {}
            for name, Y in (('X', X), ('Xres', Xres)):
                ev[name] = signed(Y, ops, iC, iC2, weigh, yc)
            ph = torch.from_numpy(rng.uniform(0, 2 * np.pi, X.shape[1] // 2 + 1).astype(np.float32))
            ph[0] = 0
            N = torch.fft.irfft(torch.fft.rfft(Xres, dim=1) * torch.exp(1j * ph), n=X.shape[1], dim=1)
            ev['null'] = signed(N, ops, iC, iC2, weigh, yc)
            mid = slice(61, -61)
            ch = torch.from_numpy((yc >= lo) & (yc <= hi))
            b0 = st0[st0[:, 4] == ib]
            row = dict(batch=ib, band=[float(lo), float(hi)], found=found_band, n_st0=int(((b0[:, 1] >= lo) & (b0[:, 1] <= hi)).sum()),
                       var_X=float(X[ch][:, mid].var()), var_Xres=float(Xres[ch][:, mid].var()), var_null=float(N[ch][:, mid].var()), match_s=tm)
            for name, (t, y, amp, s) in ev.items():
                o = np.argsort(t)
                t, y, amp, s = t[o], y[o], amp[o], s[o]
                keep = (y >= lo) & (y <= hi)
                t, y, amp, s = t[keep], y[keep], amp[keep], s[keep]
                # event times are at the wTEMP convolution's peak, spike times at the template's: same convention here
                m = nearest_event(t, y, tf, yf, 10, 50.0)
                mc = nearest_event(t, y, tf + 500, yf, 10, 50.0)
                row[name] = dict(n=int(t.size), pos=int((s > 0).sum()), near=int((m >= 0).sum()), near_chance=int((mc >= 0).sum()),
                                 amp=np.quantile(amp, [.1, .5, .9, .99]).tolist() if t.size else [],
                                 iso_pos=int(((m < 0) & (s > 0)).sum()), iso_amp=np.quantile(amp[(m < 0) & (s > 0)], [.5, .9, .99]).tolist() if ((m < 0) & (s > 0)).any() else [],
                                 depth=np.histogram(y[(m < 0) & (s > 0)], bins=np.linspace(lo, hi, 11))[0].tolist())
            rows.append(row)
            print(a.ex, rnd, json.dumps({k: (v if not isinstance(v, dict) else {x: v[x] for x in ('n', 'pos', 'near', 'near_chance', 'iso_pos')}) for k, v in row.items()}), flush=True)
            (OUT / f'm6_{a.ex}.json').write_text(json.dumps({**out, f'r{rnd}': rows}, indent=1))
        out[f'r{rnd}'] = rows
        del ctc, U
    (OUT / f'm6_{a.ex}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
