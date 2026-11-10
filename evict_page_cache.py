#!/usr/bin/env python3
"""Drop this deployment's files from the Linux page cache (posix_fadvise DONTNEED).

On GB200 the GPUs' HBM is onlined as NUMA nodes (ATS), and Grace RAM (~958 GiB) is smaller
than the checkpoint, so reading the weights spills page cache into HBM. nvidia-smi and
cudaMemGetInfo count it as used, which shrinks SGLang's KV pool and unbalances the TP ranks.
Only clean cached copies of files we can open are dropped; no root, nothing else is touched.

    evict_page_cache.py [paths...]     default: both model directories
"""
import os
from pathlib import Path
import sys

DEFAULT = ['/scratch_local/user_data/yiming/models/Kimi-K3.1-Pro',
           '/scratch_local/user_data/yiming/models/Kimi-K3.1-Dflash2-Preview']


def evict(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    files = 0
    for root in sys.argv[1:] or DEFAULT:
        root = Path(root)
        for path in [root] if root.is_file() else (p for p in root.rglob('*') if p.is_file()):
            evict(path)
            files += 1
    print(f'dropped cached pages of {files} files')


if __name__ == '__main__':
    main()
