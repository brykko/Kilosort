"""Extract an exact time segment of a SpikeGLX AP binary from an rclone remote.

Reads byte ranges with `rclone cat --offset --count`, bypassing any `rclone mount`
and its VFS cache. Data are fetched as ~1 GB pieces over several parallel
streams, each piece into its own file, so an interrupted run keeps finished
pieces. The pieces are then joined and hashed in one pass, spot-checked against
the remote, and described in a JSON manifest.

Sample i of the probe has synced time (i + 1) / fs (checked against the remote
`t_synced.npy`); the segment holds every sample with t0 <= t < t1.
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PIECE_BYTES = 1_000_000_000

p = argparse.ArgumentParser()
p.add_argument('--remote-probe', required=True, help='rclone path of the probe folder')
p.add_argument('--t0', type=float, required=True)
p.add_argument('--t1', type=float, required=True)
p.add_argument('--out-dir', required=True)
p.add_argument('--streams', type=int, default=8)
args = p.parse_args()

ap_remote = args.remote_probe.rstrip('/') + '/npx.imec.ap.bin'
meta_remote = args.remote_probe.rstrip('/') + '/npx.imec.ap.meta'
ts_remote = args.remote_probe.rstrip('/') + '/t_synced.npy'


def rclone_cat(path, offset, count):
    out = subprocess.run(['rclone', 'cat', '--offset', str(offset), '--count', str(count), path],
                         capture_output=True, check=True).stdout
    if len(out) != count:
        raise IOError(f'rclone cat returned {len(out)} of {count} bytes from {path} at {offset}')
    return out


os.makedirs(args.out_dir, exist_ok=True)
meta_local = os.path.join(args.out_dir, 'source.ap.meta')
subprocess.run(['rclone', 'copyto', meta_remote, meta_local], check=True)
meta = dict(line.split('=', 1) for line in open(meta_local).read().splitlines() if '=' in line)
fs = float(meta['imSampRate'])
n_chan = int(meta['nSavedChans'])
src_bytes = int(meta['fileSizeBytes'])
sample_bytes = 2 * n_chan

i0 = round(args.t0 * fs) - 1
i1 = round(args.t1 * fs) - 1
n_samples = i1 - i0
byte0 = i0 * sample_bytes
total = n_samples * sample_bytes
if i0 < 0 or byte0 + total > src_bytes:
    sys.exit(f'Segment {args.t0}-{args.t1} s lies outside the source file')

# Confirm the sample/time convention against the synced timestamps (two 8-byte reads).
head = rclone_cat(ts_remote, 0, 10)
hdr_len = 10 + struct.unpack('<H', head[8:10])[0] if head[6] == 1 else 12 + struct.unpack('<I', rclone_cat(ts_remote, 8, 4))[0]
t_first = struct.unpack('<d', rclone_cat(ts_remote, hdr_len + 8 * i0, 8))[0]
t_last = struct.unpack('<d', rclone_cat(ts_remote, hdr_len + 8 * (i1 - 1), 8))[0]
if not (abs(t_first - args.t0) < 0.5 / fs and t_last < args.t1 and args.t1 - t_last < 1.5 / fs):
    sys.exit(f'Timestamp check failed: t[i0]={t_first}, t[i1-1]={t_last}')

stem = f'ap_{args.t0:g}-{args.t1:g}s'
final = os.path.join(args.out_dir, stem + '.bin')
piece_dir = os.path.join(args.out_dir, stem + '.pieces')
if os.path.exists(final):
    sys.exit(f'{final} already exists')
os.makedirs(piece_dir, exist_ok=True)

piece = PIECE_BYTES - PIECE_BYTES % sample_bytes
pieces = [(k, k * piece, min(piece, total - k * piece)) for k in range(-(-total // piece))]
piece_path = lambda k: os.path.join(piece_dir, f'{k:04d}.bin')
pending = [pc for pc in pieces if not (os.path.exists(piece_path(pc[0])) and os.path.getsize(piece_path(pc[0])) == pc[2])]
done_start = total - sum(pc[2] for pc in pending)
free = shutil.disk_usage(args.out_dir).free
if free < 2 * total - done_start + 5e9:
    sys.exit(f'Not enough free space: need {(2 * total - done_start) / 1e9:.1f} GB + 5 GB margin, have {free / 1e9:.1f} GB')

print(f'Source  {ap_remote}')
print(f'Segment {args.t0:g}-{args.t1:g} s = samples [{i0}, {i1}), {n_chan} channels')
print(f'Bytes   {byte0} + {total} ({total / 1e9:.2f} GB) -> {final}')
print(f'{len(pieces)} pieces, {len(pieces) - len(pending)} already done, {args.streams} parallel streams')


def fetch(pc):
    k, off, count = pc
    tmp = piece_path(k) + '.tmp'
    for attempt in range(6):
        with open(tmp, 'wb') as f:
            proc = subprocess.run(['rclone', 'cat', '--offset', str(byte0 + off), '--count', str(count), ap_remote],
                                  stdout=f, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.getsize(tmp) == count:
            os.replace(tmp, piece_path(k))
            return
        print(f'\nPiece {k} attempt {attempt + 1} failed (exit {proc.returncode}, {os.path.getsize(tmp)} of {count} bytes): '
              f'{proc.stderr.decode(errors="replace").strip()[-300:]}')
        time.sleep(10 * (attempt + 1))
    raise IOError(f'Piece {k} failed 6 times; rerun the same command to resume')


def downloaded():
    n = 0
    for k, _, count in pieces:
        for path in (piece_path(k), piece_path(k) + '.tmp'):
            if os.path.exists(path):
                n += min(os.path.getsize(path), count)
                break
    return n


t_start = time.time()
with ThreadPoolExecutor(args.streams) as pool:
    futures = [pool.submit(fetch, pc) for pc in pending]
    while not all(fu.done() for fu in futures):
        time.sleep(2)
        got = downloaded()
        rate = (got - done_start) / max(time.time() - t_start, 1e-9)
        eta = (total - got) / max(rate, 1)
        print(f'\r{got / 1e9:7.2f} / {total / 1e9:.2f} GB  {100 * got / total:5.1f}%  '
              f'{rate / 1e6:6.1f} MB/s  ETA {int(eta // 60):3d} min {int(eta % 60):02d} s   ', end='', flush=True)
    for fu in futures:
        fu.result()
print(f'\nDownloaded in {(time.time() - t_start) / 60:.1f} min. Joining pieces and computing SHA-256...')

part = final + '.part'
h = hashlib.sha256()
with open(part, 'wb') as out:
    for k, _, count in pieces:
        with open(piece_path(k), 'rb') as f:
            while buf := f.read(64 * 2**20):
                h.update(buf)
                out.write(buf)
if os.path.getsize(part) != total:
    sys.exit(f'Joined file has {os.path.getsize(part)} bytes, expected {total}')

print('Spot-checking 20 random 1 MB blocks against the remote...')
rng = random.Random(0)
checks = []
with open(part, 'rb') as f:
    for _ in range(20):
        off = rng.randrange(0, total - 2**20)
        f.seek(off)
        ok = f.read(2**20) == rclone_cat(ap_remote, byte0 + off, 2**20)
        checks.append({'local_offset': off, 'match': ok})
        if not ok:
            sys.exit(f'Spot check FAILED at local offset {off}; {part} kept for inspection')
os.replace(part, final)
shutil.rmtree(piece_dir)

manifest = {
    'source_remote': ap_remote,
    'source_bytes': src_bytes,
    'sample_rate_hz': fs,
    'n_saved_channels': n_chan,
    'dtype': 'int16, interleaved by sample',
    'time_convention': 'synced time of sample i is (i + 1) / fs; probe_1 is the sync master',
    'requested_time_s': [args.t0, args.t1],
    'sample_range_half_open': [i0, i1],
    'synced_time_first_last_s': [t_first, t_last],
    'source_byte_offset': byte0,
    'bytes': total,
    'file': os.path.basename(final),
    'sha256': h.hexdigest(),
    'spot_checks': checks,
    'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
}
with open(os.path.join(args.out_dir, stem + '.manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'Done: {final}\nSHA-256 {h.hexdigest()}')
