# Deployment status — 2026-11-10

Kimi K3.1 Pro, TP16 · DCP16 across c04 (rank 0), c11, c13 and c14. Not serving yet: the
environment is built and replicated; the weights have not been downloaded (waiting for the
user's go-ahead, per the brief).

| Stage | State |
|---|---|
| 0. Pre-flight | passed |
| Environment (venv) | built on c04, replicated and verified on all four nodes |
| Weights | not started (needs approval) |
| 1. Baseline launch | not started |
| 2. DFLASH | not started |
| 3. HiCache | not started |

## Pre-flight (2026-11-10 17:54 UTC)

`check_rack.py` on c04, c11, c13-c14 (reports in `.runs/preflight/`): all four nodes
HEALTHY, 72/72 NVLinks, GPU recovery action None, one fabric domain
(`dedfac56-88ba-46f8-9939-7b7dc85c0dc4`, clique 32766). All 16 GPUs at 0 MiB with no
compute processes; no other users' processes or sessions (`who` showed only this session).
`nvidia-imex` active everywhere (c14 exposes only channel0, as noted in the brief).
`/scratch_local` free: c04 11 TB, c11 11 TB, c13 13 TB, c14 8.2 TB.

Differences from the brief:

- Kernel is `6.6.63.bsk.2-gb.1-rc4-64k-arm64` on the nodes checked (c04, c14), not
  6.14.0-1015-nvidia-64k; the userland is Debian 12 (glibc 2.36) although os-release says
  Ubuntu 22.04. Driver 580.167.08 and gcc-12 are as described.
- tmux, like rsync and uv, exists only in `~/.local/bin` (conda-forge prefix
  `~/.local/opt/sysutils`, not `~/.local/tools` as glm.deploy's notes say).
- c11 has only uv CPython 3.12.13. Its copy, and c13's, are byte-identical to c04's
  (excluding bytecode), so they were left untouched.

## c14 provisioning

Copied from c04 with one `tar | ssh` stream into the empty scratch home (refusing to
overwrite): `~/.local/opt/sysutils` (rsync 3.4.4, tmux 3.8, from conda-forge), `uv`, `uvx`,
`micromamba` and `~/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu`. hf, fish and
claude were not installed there; the deployment does not need them.

## Engine source

- `kimi-k3.1-day-0` exists (`8b180ca9e07e`, 2026-11-09); `kimi-k31-pro` does not. The venv
  is built from `8b180ca9e07e`.
- The cookbook's measured image was built from `9ab3c28176` plus one unmerged change, with
  FlashInfer 0.7.2. That commit is not reachable from `main` or any `kimi*` branch (likely
  force-pushed away), so it cannot be reproduced. Since then the branch bumped FlashInfer to
  0.7.3, sgl-deep-gemm 0.2.3.post2 -> 0.2.7 and sgl-flash-mla 1.0.0 -> 1.0.4 (the k31
  Dockerfile's nightly base, 2026-10-24, still had the old pins). The branch's pins are used.
- PR #48178 (route DFLASH target verify to cutedsl_mla under DCP) is not merged into the
  branch, but it targets the Kimi-K3 override path. The branch's K3.1 DCP override already
  forces `speculative_draft_attention_backend=cutedsl_mla` and `speculative_attention_mode=decode`
  for DFLASH (`arg_groups/model_overrides/kimi_k3.py`, `_kimi_k31_dcp_overrides`).
- Every flag the brief marked untested exists: `--dcp-size`, `--mamba-ssm-dtype`,
  `--enable-linear-replayssm-spec`, `--speculative-use-block-verification`,
  `--speculative-draft-attention-backend`, `--speculative-draft-kv-cache-dtype`,
  `--hicache-mem-layout`, `--mamba-max-states-per-path`. K3.1 DCP also needs the fused v2 DSA
  top-k (`SGLANG_OPT_USE_TOPK_V2`, `SGLANG_DSA_FUSE_TOPK`, both default on).

## Environment

`build_venv.sh` reproduces the Python layer of `docker/k31.Dockerfile` (base nightly solve plus
the K3.1 steps) without containers. Decisions:

- CPython 3.12.13 (the Dockerfile's `PYTHON_VERSION=3.12`; ds.deploy used 3.11).
- `humming-kernels[cu13]` leaves nvcc and cccl unbounded, so the plain solve picked CUDA 13.5
  compilers next to torch's CUDA 13.0 runtime. `constraints-cuda130.txt` keeps nvcc, crt,
  nvvm, nvjitlink and cccl on 13.0 (13.0.88 / 13.0.85), matching torch 2.14.1+cu130, the
  image's CUDA 13.0.3 and driver 580. No other package changed.
- `UV_LINK_MODE=copy`: uv's default hardlinks would have carried the in-place FlashInfer
  DynBlock edits into uv's cache. The first build attempt used hardlinks and was stopped
  before the patch step ran.
- SGLang installed non-editable from the checkout with all five Rust extensions built by
  Rust 1.92.0 (conda-forge, `~/.local/opt/rust`, c04 only). In a non-checkout install the
  loader trusts the bundled modules; launches also set `SGLANG_RUST_BUILD_MODE=never` as the
  image does, so workers never need Cargo.
- flashinfer-jit-cache 0.7.3 is a meta-package over per-architecture providers; only
  `flashinfer-jit-cache-sm100a` is installed (GB200 is compute 10.0). The image would pull all
  eight. The DynBlock patch (`DynBlockKernelMaxNumExperts` 512 -> 1024, still 512 in 0.7.3) and
  the rebuild of `fused_moe_trtllm_sm100` and `trtllm_gen_routing` were applied.
- Not installed (not needed for this deployment): the Prims-TS overlay (opt-in DSA backend;
  its script requires FlashInfer 0.7.2), Dynamo, Mooncake, MSCCL++, NIXL, the `all` extra
  (diffusion etc.), dev tools.

Two more gaps surfaced while building, both fixed in `build_venv.sh`:

- `--no-deps` on the jit-cache meta-package (as the Dockerfile does) installs no provider, so
  the DynBlock rebuild found nothing to rebuild. The sm100a provider is now installed with it.
- The pip CUDA tree (`nvidia/cu13`) has `lib/` with versioned sonames only, while FlashInfer's
  JIT links `-L$CUDA_HOME/lib64 -lcudart`. The build adds `lib64 -> lib` and unversioned
  `lib*.so` links inside the venv (outside any wheel's RECORD), the layout a full toolkit has;
  serve-time JIT builds on every node rely on the same links.

Result (`.runs/build/build_venv.log`; earlier attempts kept beside it): sglang
`0.5.24.dev1106+g8b180ca9e`, 17 GB venv. `verify_venv.py --gpu` passes on c04, c11, c13 and
c14: all pins, `KimiK31ForConditionalGeneration` registered, bundled Rust TreeCore,
DynBlock header at 1024 and both modules rebuilt for sm_100a and loadable through tvm_ffi,
nvcc 13.0.88, torch CUDA 13.0 on driver 580 (no compatibility errors), 4 GB200s at compute
10.0, a bf16 matmul, NCCL 2.30.7. `replicate_venv.sh` copied it in 43 s with rsync; c11 and
c13 already had the identical uv Python, c14 got c04's earlier. `kimictl preflight` now
blocks only on the missing verified weights.

## Endpoint

`serving.json` binds `10.78.202.32:30000` (c04's bond0) rather than the brief's `0.0.0.0`:
every c04 interface is private, but `0.0.0.0` would also listen on the 10.78.203.x management
network. Nothing is published outside the cluster. The API key was generated with
`kimictl init-key` into `.secrets/` (mode 0600; ds.deploy's secrets are not reused).

## Weights (pending approval)

Manifests fetched from the HF API with the brief's token (anonymous calls were rate-limited
on the shared egress IP, as was the GitHub API):

- moonshotai/Kimi-K3.1-Pro @ d9e99b0792a5e93b78ea9bea62a68167e36ddd9e: 112 files, 96
  safetensors, 1.561 TB (1.420 TiB), not gated.
- RadixArk/Kimi-K3.1-Dflash2-Preview @ ea257b38a8ee755b65287bb8ccd0938b519ce26b: 5 files,
  1 safetensors, 5.57 GB, not gated.

Plan: `stage_weights.sh download` on c04 (flat `--local-dir`, hf_xet), verify, then
`distribute` with rsync to c11, c13 and c14 over bond0 (8 parallel streams per node, AES-GCM),
then `verify` on all four nodes.
