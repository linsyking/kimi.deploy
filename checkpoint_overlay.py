#!/usr/bin/env python3
"""Serve pqhaz's Kimi K3.1 Pro abliteration patch without copying or editing the official shards.

The patch is XOR deltas for 278 BF16 tensors (attention o_proj, dense and shared-expert down_proj,
routed_expert_up_proj) spread over 93 of the 96 shards, so a patched checkpoint would be a 1.56 TB
copy per node. As ds.deploy does for its Heretic weights, the official checkpoint stays the only
one: `build` writes just the 278 patched tensors (29.7 GB) into an overlay safetensors file,
checking every delta, original and result against the pinned manifest's SHA-256s, and install()
substitutes them by name in SGLang's weight loader, before TP sharding. Quantized experts, vision,
tokenizer, encoder and the DFLASH draft load unchanged.

    checkpoint_overlay.py download     on c04: the patch's 14.7 GB of compressed deltas
    checkpoint_overlay.py build        on c04 (venv Python): deltas + official shards -> overlay
    checkpoint_overlay.py distribute   on c04: rsync the overlay to c11, c13, c14 and verify it there
    checkpoint_overlay.py verify       hash the overlay on this node and record it as verified
    checkpoint_overlay.py check        the cheap check kimictl runs before a start (no hashing)

Bulk reads and writes run bound to Grace RAM (numactl --membind), so their page cache cannot
spill into the HBM of GPUs a running server is using, and they drop the pages they cached."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = 'pqhaz/Kimi-K3.1-Pro-abliteration-patch-20261105'
REVISION = 'f9c1b234cd8ab798a1d042e59de8bf7de938f344'
MANIFEST_SHA256 = '6899803f681c45e00eca70212b058839fac148d29f5892b250c8e9bbc9608c55'
# pqhaz made the patch against 6d0a0202; that revision's 96 shards and index are byte-identical
# to this one's (only encoding_k3.py differs), and build checks every original tensor's SHA-256.
BASE_REVISION = 'd9e99b0792a5e93b78ea9bea62a68167e36ddd9e'
TENSORS = 278
DIRECTORY = ROOT / '.runs/abliteration' / REVISION
SOURCE = DIRECTORY / 'source'  # pqhaz's files as downloaded; c04 only
OVERLAY = DIRECTORY / 'overlay.safetensors'
LOCAL = Path('/scratch_local/user_data/yiming/.local')
HF = LOCAL / 'bin/hf'
RSYNC = LOCAL / 'bin/rsync'
NUMACTL = LOCAL / 'opt/sysutils/bin/numactl'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
CHUNK = 16 << 20


def safetensors_header(path):
    """(header, offset of the data section)."""
    with Path(path).open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        if not 0 < length <= 64 << 20:
            raise ValueError(f'invalid safetensors header length: {path}')
        return json.loads(handle.read(length)), 8 + length


def manifest(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError(f'{path} is not the pinned patch manifest')
    return json.loads(raw)


def base_layout(base, patch):
    """Check that every patched tensor sits in the official checkpoint where the manifest says."""
    base = Path(base)
    verified = json.loads((base / '.kimi-verified.json').read_text())
    if verified.get('revision') != BASE_REVISION:
        raise ValueError(f'the patch needs the verified official revision {BASE_REVISION[:12]}')
    index = (base / 'model.safetensors.index.json').read_bytes()
    if hashlib.sha256(index).hexdigest() != patch['index_sha256']:
        raise ValueError('the checkpoint index differs from the one the patch was made for')
    weight_map = json.loads(index)['weight_map']
    headers = {}
    for change in patch['changes']:
        name, shard = change['name'], change['shard']
        if weight_map.get(name) != shard:
            raise ValueError(f'{name} is not in {shard}')
        if shard not in headers:
            headers[shard] = safetensors_header(base / shard)
        header, start = headers[shard]
        entry = header.get(name, {})
        begin, end = entry.get('data_offsets', (0, 0))
        if (entry.get('dtype'), entry.get('shape'), change['dtype']) != ('BF16', change['shape'], 'BF16') or (
                start + begin, end - begin) != (change['offset'], change['bytes']) or (
                change['bytes'] != 2 * math.prod(change['shape'])):
            raise ValueError(f'{name}: the checkpoint layout differs from the manifest')


def overlay_header(patch):
    """The overlay's exact header bytes (the tensors in manifest order) and its data size."""
    header, offset = {'__metadata__': {'repo': REPO, 'revision': REVISION, 'manifest_sha256': MANIFEST_SHA256,
                                       'base_revision': BASE_REVISION}}, 0
    for change in patch['changes']:
        header[change['name']] = {'dtype': 'BF16', 'shape': change['shape'],
                                  'data_offsets': [offset, offset + change['bytes']]}
        offset += change['bytes']
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    return struct.pack('<Q', len(raw)) + raw, offset


def overlay_layout(patch, overlay):
    """[(change, absolute offset)] once the overlay's header and size match the manifest."""
    header, size = overlay_header(patch)
    with Path(overlay).open('rb') as handle:
        if handle.read(len(header)) != header:
            raise ValueError(f'{overlay} does not hold the pinned manifest\'s tensors')
    if Path(overlay).stat().st_size != len(header) + size:
        raise ValueError(f'{overlay} has the wrong size')
    layout, offset = [], len(header)
    for change in patch['changes']:
        layout.append((change, offset))
        offset += change['bytes']
    return layout


def read_range(fd, offset, length):
    data = bytearray()
    while len(data) < length:
        chunk = os.pread(fd, min(CHUNK, length - len(data)), offset + len(data))
        if not chunk:
            raise ValueError('unexpected end of file')
        data += chunk
    return bytes(data)


def range_sha256(path, offset, length):
    hasher, done = hashlib.sha256(), 0
    fd = os.open(path, os.O_RDONLY)
    try:
        while done < length:
            chunk = os.pread(fd, min(CHUNK, length - done), offset + done)
            if not chunk:
                break
            hasher.update(chunk)
            done += len(chunk)
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    return hasher.hexdigest()


def file_identity(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns]


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def build(base, source=SOURCE, overlay=OVERLAY):
    """Write the patched tensors; every delta, original tensor and result is hash-checked."""
    import numpy
    import zstandard
    base, overlay = Path(base), Path(overlay)
    patch = manifest(source / 'patch/manifest.json')
    base_layout(base, patch)
    header, _ = overlay_header(patch)
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.with_suffix('.verified.json').unlink(missing_ok=True)
    partial = overlay.with_name(overlay.name + '.partial')
    partial.unlink(missing_ok=True)
    decompressor, started, written = zstandard.ZstdDecompressor(), time.monotonic(), 0
    with partial.open('xb') as out:
        out.write(header)
        for number, change in enumerate(patch['changes'], 1):
            if not re.fullmatch(r'\d{4}\.xor\.zst', change['xor_file']):
                raise ValueError(f'unexpected delta file name {change["xor_file"]!r}')
            delta_path = source / 'patch' / change['xor_file']
            compressed = delta_path.read_bytes()
            if hashlib.sha256(compressed).hexdigest() != change['xor_sha256']:
                raise ValueError(f'{delta_path} does not match the manifest')
            if zstandard.frame_content_size(compressed) not in (-1, change['bytes']):
                raise ValueError(f'{delta_path} decompresses to the wrong size')
            delta = decompressor.decompress(compressed, max_output_size=change['bytes'])
            fd = os.open(base / change['shard'], os.O_RDONLY)
            try:
                original = read_range(fd, change['offset'], change['bytes'])
                os.posix_fadvise(fd, change['offset'], change['bytes'], os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            if len(delta) != change['bytes'] or hashlib.sha256(original).hexdigest() != change['before_sha256']:
                raise ValueError(f'{change["name"]}: the delta or the official tensor does not match the manifest')
            patched = numpy.bitwise_xor(numpy.frombuffer(original, numpy.uint8),
                                        numpy.frombuffer(delta, numpy.uint8)).tobytes()
            if hashlib.sha256(patched).hexdigest() != change['after_sha256']:
                raise ValueError(f'{change["name"]}: the patched tensor does not match the manifest')
            out.write(patched)
            written += len(patched)
            with delta_path.open('rb') as handle:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            if number % 16 == 0 or number == len(patch['changes']):
                out.flush()
                os.fsync(out.fileno())
                os.posix_fadvise(out.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                print(f'[{int(time.monotonic() - started)} s] {number}/{len(patch["changes"])} tensors, '
                      f'{written / 1e9:.1f} GB', flush=True)
    partial.chmod(0o444)
    partial.replace(overlay)
    # The copy kimictl and the workers read; verify checks it against the pin again.
    temporary = overlay.parent / 'manifest.json.tmp'
    temporary.write_bytes((source / 'patch/manifest.json').read_bytes())
    temporary.replace(overlay.parent / 'manifest.json')
    verify(base, overlay)


def verify(base, overlay=OVERLAY):
    """Hash every overlay tensor against the manifest and record this file as verified."""
    overlay = Path(overlay)
    patch = manifest(overlay.parent / 'manifest.json')
    base_layout(base, patch)
    marker = overlay.with_suffix('.verified.json')
    marker.unlink(missing_ok=True)
    layout = overlay_layout(patch, overlay)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        digests = list(pool.map(lambda item: range_sha256(overlay, item[1], item[0]['bytes']), layout))
    bad = [change['name'] for (change, _), digest in zip(layout, digests) if digest != change['after_sha256']]
    if bad:
        raise ValueError(f'{len(bad)} overlay tensors do not match the manifest, first {bad[0]}')
    write_json(marker, {'repo': REPO, 'revision': REVISION, 'manifest_sha256': MANIFEST_SHA256,
                        'base_revision': BASE_REVISION, 'tensors': len(layout), 'file': file_identity(overlay),
                        'host': socket.gethostname().split('.')[0],
                        'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')})
    print(f'{socket.gethostname().split(".")[0]}: {len(layout)} overlay tensors match the manifest '
          f'({int(time.monotonic() - started)} s); wrote {marker}', flush=True)


def check(base, overlay=OVERLAY):
    """Before every start and in every rank's loader: the pinned manifest, the official layout, the
    overlay's header and size, and a verified marker for this very file. Hashing is verify's job."""
    overlay = Path(overlay)
    patch = manifest(overlay.parent / 'manifest.json')
    base_layout(base, patch)
    overlay_layout(patch, overlay)
    try:
        marker = json.loads(overlay.with_suffix('.verified.json').read_text())
    except FileNotFoundError:
        raise ValueError(f'{overlay} is not verified on this node; run checkpoint_overlay.py verify') from None
    if marker.get('manifest_sha256') != MANIFEST_SHA256 or marker.get('file') != file_identity(overlay):
        raise ValueError(f'{overlay} changed since it was verified; run checkpoint_overlay.py verify')
    return {'repo': REPO, 'revision': REVISION, 'manifest_sha256': MANIFEST_SHA256,
            'overlay': str(overlay.resolve()), 'base': str(Path(base).resolve()),
            'tensor_names': [change['name'] for change in patch['changes']]}


def replaced(weights, overlay, names, seen):
    """Keep the checkpoint's names and order; swap in the overlay's tensor for each patched name."""
    from safetensors import safe_open
    with safe_open(str(overlay), framework='pt', device='cpu') as replacements:
        for name, weight in weights:
            if name in names:
                if name in seen:
                    raise RuntimeError(f'duplicate checkpoint tensor: {name}')
                replacement = replacements.get_tensor(name)
                if replacement.shape != weight.shape or replacement.dtype != weight.dtype:
                    raise RuntimeError(f'runtime tensor mismatch: {name}')
                seen.add(name)
                yield name, replacement
            else:
                yield name, weight


def install(overlay, base, audit_dir):
    from functools import wraps
    from sglang.srt.model_loader.loader import DefaultModelLoader
    original = DefaultModelLoader.load_weights_only
    if getattr(original, '_kimi_checkpoint_overlay', False):
        return

    @wraps(original)
    def load(model, weights, target_device):
        # The DFLASH draft passes through this loader too; only the target's tensors are replaced.
        if type(model).__name__ != 'KimiK31ForConditionalGeneration':
            return original(model, weights, target_device)
        # Failures are recorded, not raised: a rank that raises here takes the start down and leaves
        # its peers to a forced stop, while a short record fails `kimictl weights`, so modelctl
        # stops the server gracefully and restores the previous weights.
        try:
            metadata = check(base, overlay)
        except (OSError, ValueError, KeyError) as error:
            metadata = {'error': str(error), 'tensor_names': []}
        names, seen = set(metadata.pop('tensor_names')), set()
        result = original(model, replaced(weights, overlay, names, seen) if names else weights, target_device)
        import torch.distributed as distributed
        rank = distributed.get_rank() if distributed.is_initialized() else 0
        destination = Path(audit_dir)
        destination.mkdir(parents=True, exist_ok=True)
        record = dict(metadata, rank=rank, pid=os.getpid(), loaded_at=time.time(), tensor_count=len(seen),
                      model_class=type(model).__name__)
        write_json(destination / f'rank-{rank}-{os.getpid()}.json', record)
        print(f'Kimi checkpoint overlay: rank {rank}, {len(seen)}/{len(names)} tensors loaded'
              + ('' if names and seen == names else f' (FAILED: {metadata.get("error", "incomplete")})'), flush=True)
        return result

    load._kimi_checkpoint_overlay = True
    DefaultModelLoader.load_weights_only = staticmethod(load)


def require_head(cfg):
    if socket.gethostname().split('.')[0] != cfg['nodes'][0]['hostname']:
        raise RuntimeError(f'run this on {cfg["nodes"][0]["name"]}')


def bind_to_grace():
    """Re-run this command with its memory, page cache included, bound to the Grace NUMA nodes."""
    if os.environ.get('KIMI_OVERLAY_MEMBIND') != '1':
        nodes = Path('/sys/devices/system/node/has_cpu').read_text().strip()
        command = [str(NUMACTL), f'--membind={nodes}', sys.executable, '-I', *sys.argv]
        os.execve(command[0], command, dict(os.environ, KIMI_OVERLAY_MEMBIND='1'))


def download(cfg):
    require_head(cfg)
    patch = manifest(SOURCE / 'patch/manifest.json')
    env = dict(os.environ, HF_HOME='/scratch_local/user_data/yiming/.cache/huggingface',
               HF_HUB_DISABLE_TELEMETRY='1', HF_XET_HIGH_PERFORMANCE='1')
    subprocess.run([str(HF), 'download', REPO, '--revision', REVISION, '--include', 'patch/*.xor.zst',
                    '--local-dir', str(SOURCE), '--max-workers', '16'], env=env, check=True)
    sizes = {change['xor_file']: change['compressed_bytes'] for change in patch['changes']}
    wrong = [name for name, size in sizes.items()
             if not (SOURCE / 'patch' / name).is_file() or (SOURCE / 'patch' / name).stat().st_size != size]
    if wrong:
        raise RuntimeError(f'{len(wrong)} delta files are missing or the wrong size, first {wrong[0]}')
    print(f'Downloaded {len(sizes)} delta files ({sum(sizes.values()) / 1e9:.2f} GB); build checks their hashes.')


def distribute(cfg):
    """Copy the overlay to the worker nodes, then verify it on each of them."""
    require_head(cfg)
    check(cfg['model_path'])

    def push(node):
        remote = [*SSH, node['host']]
        subprocess.run([*remote, shlex.join(['mkdir', '-p', str(DIRECTORY)])], check=True, timeout=30)
        nodes = subprocess.run([*remote, 'cat /sys/devices/system/node/has_cpu'], check=True, text=True,
                               capture_output=True, timeout=30).stdout.strip()
        rsync = [str(RSYNC), '-a', '--partial-dir=.rsync-partial',
                 # AES-GCM uses the Grace cores' crypto extensions (as stage_weights.sh does).
                 '-e', shlex.join([*SSH, '-c', 'aes128-gcm@openssh.com']),
                 f'--rsync-path={NUMACTL} --membind={nodes} {RSYNC}']
        subprocess.run([*rsync, str(ROOT / 'checkpoint_overlay.py'), f'{node["host"]}:{ROOT}/'], check=True)
        subprocess.run([*rsync, str(OVERLAY), str(DIRECTORY / 'manifest.json'), f'{node["host"]}:{DIRECTORY}/'],
                       check=True)
        subprocess.run([*remote, shlex.join(['/usr/bin/python3', '-I', str(ROOT / 'checkpoint_overlay.py'),
                                             'verify'])], check=True)

    with concurrent.futures.ThreadPoolExecutor(len(cfg['nodes']) - 1) as pool:
        list(pool.map(push, cfg['nodes'][1:]))
    print(f'The overlay is verified on {", ".join(node["name"] for node in cfg["nodes"])}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['download', 'build', 'distribute', 'verify', 'check'])
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'serving.json').read_text())
    if args.action == 'check':
        metadata = check(cfg['model_path'])
        print(json.dumps(dict(metadata, tensor_names=len(metadata['tensor_names']))))
        return 0
    bind_to_grace()
    if args.action == 'download':
        download(cfg)
    elif args.action == 'build':
        require_head(cfg)
        build(cfg['model_path'])
    elif args.action == 'distribute':
        distribute(cfg)
    else:
        verify(cfg['model_path'])
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
