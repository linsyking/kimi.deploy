#!/usr/bin/env python3
"""Verify a model directory against its Hugging Face manifest and record the result.

Every file in manifests/<name>.manifest.json must exist with the listed size: LFS files by
their sha256, the rest by their git blob sha1. On success this writes .kimi-verified.json
into the directory, which kimictl's preflight requires. Stdlib only; runs with /usr/bin/python3.

    verify_weights.py /scratch_local/user_data/yiming/models/Kimi-K3.1-Pro
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent
MARKER = '.kimi-verified.json'
CHUNK = 16 << 20


def digest(path, row):
    if row['sha256']:
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.sha1(b'blob %d\0' % row['size'])
    with open(path, 'rb') as handle:
        while chunk := handle.read(CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def check(directory, row):
    path = directory / row['path']
    if not path.is_file():
        return row['path'], 'missing'
    if path.stat().st_size != row['size']:
        return row['path'], f'size {path.stat().st_size} != {row["size"]}'
    expected = row['sha256'] or row['blob_id']
    found = digest(path, row)
    return row['path'], None if found == expected else f'hash {found[:12]} != {expected[:12]}'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--manifest', type=Path, help='default: manifests/<directory name>.manifest.json')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    manifest_path = args.manifest or ROOT / 'manifests' / f'{args.directory.name}.manifest.json'
    manifest = json.loads(manifest_path.read_text())
    marker = args.directory / MARKER
    marker.unlink(missing_ok=True)
    rows = sorted(manifest['files'], key=lambda row: -row['size'])
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(check, [args.directory] * len(rows), rows))
    failures = [(path, error) for path, error in results if error]
    expected = {row['path'] for row in rows}
    extra = sorted(str(path.relative_to(args.directory)) for path in args.directory.rglob('*')
                   if path.is_file() and not path.relative_to(args.directory).parts[0] in ('.cache', MARKER)
                   and str(path.relative_to(args.directory)) not in expected)
    for path, error in failures:
        print(f'FAIL {path}: {error}')
    for path in extra:
        print(f'note: file not in manifest: {path}')
    total = sum(row['size'] for row in rows)
    host = socket.gethostname().split('.')[0]
    if failures:
        print(f'{host}: {len(failures)} of {len(rows)} files failed in {args.directory}')
        return 1
    record = {'repo': manifest['repo'], 'revision': manifest['revision'], 'files': len(rows), 'bytes': total,
              'host': host, 'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=1) + '\n')
    os.replace(temporary, marker)
    print(f'{host}: all {len(rows)} files ({total / 1e12:.3f} TB) match {manifest["repo"]}@{manifest["revision"][:12]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
