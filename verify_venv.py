#!/usr/bin/env python3
"""Check .venvs/sglang on this node: pins, the K3.1 model class, the compiled K3.1 parts that
fall back silently (Rust TreeCore, FlashInfer DynBlock), and the CUDA 13.0 toolchain.
Run with the venv's interpreter; --gpu adds a CUDA/NCCL smoke test on GPU 0."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path

ROOT = Path('/scratch_local/user_data/yiming/kimi.deploy')
VENV = Path(os.environ.get('KIMI_VENV', ROOT / '.venvs/sglang'))
SITE = Path(sysconfig.get_paths()['purelib'])
CUDA = SITE / 'nvidia/cu13'
PINS = {
    'torch': '2.14.1+cu130', 'flashinfer-python': '0.7.3', 'flashinfer-cubin': '0.7.3',
    'flashinfer-jit-cache': '0.7.3+cu130', 'flashinfer-jit-cache-sm100a': '0.7.3+cu130',
    'sglang-kernel': '0.4.10', 'sgl-deep-gemm': '0.2.7',
    'sgl-flash-mla': '1.0.4', 'sgl-deep-ep': '0.1.2.post1', 'nvidia-nccl-cu13': '2.30.7',
    'transformers': '5.19.0', 'compressed-tensors': '0.18.0', 'nvidia-cuda-nvcc': '13.0.88',
    'nvidia-cuda-runtime': '13.0.96', 'pypdfium2': '5.15.2',  # pypdfium2: documents.py
}
DYNBLOCK_MODULES = ('fused_moe_trtllm_sm100', 'trtllm_gen_routing')

failures = []


def check(name, ok, detail=''):
    print(f'{"ok  " if ok else "FAIL"} {name}{": " + detail if detail else ""}')
    if not ok:
        failures.append(name)


def check_pins():
    for package, expected in PINS.items():
        try:
            found = metadata.version(package)
        except metadata.PackageNotFoundError:
            found = 'missing'
        check(f'{package}=={expected}', found == expected, '' if found == expected else f'found {found}')


def check_sglang():
    build = json.loads((VENV / 'kimi-build.json').read_text())
    check('sglang build record', True, f'{build["sglang_branch"]} @ {build["sglang_commit"][:12]}')
    import sglang
    check('sglang import', Path(sglang.__file__).is_relative_to(SITE), sglang.__file__)
    from sglang.srt.models.registry import ModelRegistry
    check('KimiK31ForConditionalGeneration registered',
          'KimiK31ForConditionalGeneration' in ModelRegistry.get_supported_archs())
    from sglang.srt.mem_cache.rust_tree_core import extension
    check('Rust TreeCore (bundled)', extension.bindings.__file__.startswith(str(SITE)),
          extension.bindings.__file__)


def check_dynblock():
    header = SITE / 'flashinfer/data/include/flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh'
    match = re.search(r'DynBlockKernelMaxNumExperts = (\d+);', header.read_text())
    check('FlashInfer DynBlock header >= 896 experts', bool(match and int(match.group(1)) >= 896),
          match and match.group(1))
    digest = hashlib.sha256(header.read_bytes()).hexdigest()
    modules = [path for name in DYNBLOCK_MODULES for path in SITE.glob(f'flashinfer*/**/{name}.so')]
    check('FlashInfer DynBlock modules present', bool(modules), f'{len(modules)} found')
    import tvm_ffi
    for path in modules:
        marker = path.parent / 'k31-dynblock.json'
        record = json.loads(marker.read_text()) if marker.exists() else {}
        check(f'rebuilt {path.relative_to(SITE)}',
              record.get('header_sha256') == digest and record.get('DynBlockKernelMaxNumExperts') == 1024,
              record.get('arch', 'no marker'))
        try:
            # FlashInfer's own loader (flashinfer/jit/core.py); catches unresolved link-time symbols.
            tvm_ffi.load_module(str(path))
            check(f'loads {path.name}', True)
        except Exception as exc:
            check(f'loads {path.name}', False, str(exc))


def check_toolchain():
    out = subprocess.run([str(CUDA / 'bin/nvcc'), '--version'], capture_output=True, text=True).stdout
    check('nvcc is CUDA 13.0', 'release 13.0' in out, out.strip().splitlines()[-1] if out else 'no nvcc')


def check_gpu():
    import torch
    check('torch CUDA build 13.0', torch.version.cuda == '13.0', torch.version.cuda)
    check('CUDA available', torch.cuda.is_available())
    count = torch.cuda.device_count()
    check('4 GPUs visible', count == 4, str(count))
    capability = torch.cuda.get_device_capability(0)
    check('GB200 compute capability 10.0', capability == (10, 0), str(capability))
    a = torch.randn(1024, 1024, device='cuda:0', dtype=torch.bfloat16)
    check('bf16 matmul on GPU 0', torch.isfinite(a @ a).all().item())
    nccl = '.'.join(map(str, torch.cuda.nccl.version()))
    check('NCCL runtime 2.30.7', nccl == '2.30.7', nccl)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='also run the CUDA smoke test on GPU 0')
    args = parser.parse_args()
    check('venv interpreter', Path(sys.prefix) == VENV, sys.prefix)
    check_pins()
    check_sglang()
    check_dynblock()
    check_toolchain()
    if args.gpu:
        check_gpu()
    print(f'\n{len(failures)} failure(s)' if failures else '\nall checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
