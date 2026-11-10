# Deployment status — 2026-11-10

Kimi K3.1 Pro, TP16 · DCP16 across c04 (rank 0), c11, c13 and c14. Serving since 19:24 UTC
on http://10.78.202.32:30000/v1 (model `kimi-k3.1-pro`), stage 1 (plain decoding, 262,144
context). Every item of the brief's validation checklist passes for this stage.

| Stage | State |
|---|---|
| 0. Pre-flight | passed |
| Environment (venv) | built on c04, replicated and verified on all four nodes |
| Weights | downloaded on c04, rsynced, verified on all four nodes |
| 1. Baseline launch | serving; functional checks, stress test and bench sweep pass |
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
  for DFLASH (`arg_groups/model_overrides/kimi_k3.py`, `_kimi_k31_dcp_overrides`). The PR's bug:
  under DCP, K3's override sent DFLASH target verify to the prefill kernel (`trtllm_mla`), which
  cannot take the global causal bound that verifying several draft tokens over DCP-sharded KV
  needs; DSPARK and EAGLE3 were already routed to decode. The unfixed K3 override is registered
  for `KimiK3ForConditionalGeneration` only. Checked offline with the stage-2 flags (8 draft
  tokens, no GPU visible, `.runs/dryrun_dflash_resolved.py`): the K3.1 override dispatch declares
  `speculative_attention_mode=decode` and `cutedsl_mla`, and no later hook rewrites the mode.
  SGLang records overrides as declarations, so the server_args dump in `sglang.log` shows the raw
  input (`'speculative_attention_mode': 'prefill'`), not the resolved value.
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
c13 already had the identical uv Python, c14 got c04's earlier.

## Endpoint

`serving.json` binds `10.78.202.32:30000` (c04's bond0) rather than the brief's `0.0.0.0`:
every c04 interface is private, but `0.0.0.0` would also listen on the 10.78.203.x management
network. Nothing is published outside the cluster. The API key was generated with
`kimictl init-key` into `.secrets/` (mode 0600; ds.deploy's secrets are not reused).

## Weights

Manifests fetched from the HF API with the brief's token (anonymous calls were rate-limited
on the shared egress IP, as was the GitHub API):

- moonshotai/Kimi-K3.1-Pro @ d9e99b0792a5e93b78ea9bea62a68167e36ddd9e: 112 files, 96
  safetensors, 1.561 TB (1.420 TiB), not gated.
- RadixArk/Kimi-K3.1-Dflash2-Preview @ ea257b38a8ee755b65287bb8ccd0938b519ce26b: 5 files,
  1 safetensors, 5.57 GB, not gated.

`stage_weights.sh` downloaded both once on c04 (flat `--local-dir`, hf_xet, about 4.3 GB/s),
rsynced them to c11, c13 and c14 over bond0 (8 streams per node, AES-GCM), and verified every
file on every node against the manifests: Pro 112 files / 1,561,355,966,953 bytes and the draft
5 files / 5,569,652,493 bytes on all four nodes (`.kimi-verified.json`, 19:05 UTC). The whole
chain took about 19 minutes.

## GB200: page cache in HBM

After staging, nvidia-smi showed memory "used" on idle GPUs with no processes: up to 95.9 GiB of
page cache sat in one GPU's HBM NUMA node, unevenly spread (45.4, 9.2, 1.4 GiB on others). Why:
on GB200 each GPU's HBM is onlined as a CPU-less NUMA node (2, 10, 18, 26; ATS), Grace RAM is
958 GiB per node (the brief's 1,694 GiB counts HBM), and the kernel spills file cache into HBM
once Grace fills. nvidia-smi and `cudaMemGetInfo` count those pages as used, so SGLang would size
a smaller and per-rank uneven KV pool. Fixes, all without root:

- `evict_page_cache.py` drops the model files' cached pages (posix_fadvise DONTNEED); `start`
  runs it on every node and preflight refuses to start with 1 GiB or more of page cache in HBM.
- Every rank runs under `numactl --membind=0,1`, which keeps host allocations and page cache on
  Grace; CUDA device allocations ignore the policy. Checked before launch: a 150 GiB
  `cudaMalloc` still lands in HBM node 2, and reading the full 1.454 TiB checkpoint at 48.8 GB/s
  left no page cache in HBM. During the stage-1 load HBM nodes held 0.0 GiB of file pages.
- numactl was missing on c11; c04's conda-forge prefix `~/.local/opt/sysutils` was copied there
  (c13 and c14 already had it).

## Stage 1: baseline launch

`serving.json`: TP16, DCP16, 4 nodes, `--kv-cache-dtype fp8_e4m3`, `--mamba-ssm-dtype float32`,
`--mem-fraction-static 0.85`, context 262,144, `--max-running-requests 64`, radix cache off,
`--reasoning-parser kimi_k3 --tool-call-parser kimi_k3`, `--watchdog-timeout 3600`,
`--disable-flashinfer-autotune`, multithreaded weight loading, `SGLANG_K3_SHARDED_MOE=0`,
`NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`, `NCCL_DEBUG=INFO` (first launch only). The DCP
communication backend, symmetric memory, the MoE A2A/runner backends and the DSA backend are
left to SGLang; it resolved `dsa_topk_backend=sgl-kernel` (required by the K3.1 DCP override).

Launch (19:13 UTC, `kimictl start`): ready after 672 s. Weights loaded in 265 s on every rank
(132.9 GB per GPU), decode CUDA graphs for batch 1-64 captured in 251 s, and a 1-minute warmup
while the first requests compiled Triton KDA kernels. KV pool: 14,902,272 tokens across the 16
DCP ranks. Rank 0 has 1.8 GB less free memory than the others because SGLang's main process
keeps a CUDA context on GPU 0 for multimodal preprocessing.

- NCCL: every channel on all four nodes is `P2P/MNNVL` (506-510 channel lines per node), with no
  socket or NET transport; NVLS stays off (`NCCL_NVLS_ENABLE=0`, SGLang's default without
  `--enable-nccl-nvls`). The API key is redacted in the logged server_args.
- Followers log `[Errno 99] error while attempting to bind on address ('10.78.202.32', 30000)`:
  SGLang starts a dummy health endpoint on every node at rank 0's bind address, which only exists
  on c04. Harmless; nothing probes followers.

Functional checks (`smoke_test.py`, `.runs/smoke/20261110-112607.json`), all passing:

| Check | Result |
|---|---|
| `/health`, `/v1/models` | 200; `kimi-k3.1-pro`, max_model_len 262,144 |
| Chat, `reasoning_effort` low | "24" (360's divisors), reasoning present, 40 completion tokens |
| Chat, `reasoning_effort` max | "24", reasoning present, 86 completion tokens |
| Tool call | `get_weather({"city": "Paris", "unit": "celsius"})`, then a final answer using the tool result |
| Image input | 256x256 PNG, red left / blue right: "left: red, right: blue" |
| Needle, 127,902 prompt tokens | needle at 60% depth found; 5.6 s end to end |

Reasoning effort: a request without the field renders like `max` (the model card's default) and
`high` works, but **`medium` fails with HTTP 500** (`Unsupported thinking_effort='medium'`). The
branch maps `medium` to a native K3.1 tier (`_KIMI_K31_EFFORT_TIERS` in `serving_chat.py`),
while this checkpoint's `encoding_k3.py` accepts only low, high and max, which is also what the
model card documents. The server stays healthy. Not patched; clients should not send `medium`.

Decode-hang test (issue #39568): 32 concurrent requests of 8K input / 1K output right after
startup (`.runs/bench/stage1-stress-c32.log`). All 32 completed in 30.1 s with no stall after the
first decode step: decode ran at about 1,650 tok/s, TPOT mean 23.8 ms / p90 27.1 ms, TTFT mean
5.7 s (all 32 prefills arrived at once; prefill peaked at 46K tok/s). No errors in any log.

Benchmarks (`bench.sh stage1`, `.runs/bench/stage1-c*.{log,jsonl}`): `sglang.benchmark.serving`
on c04 against `/generate`, random dataset from ShareGPT text with fixed 8,192 input and 1,024
output tokens, 4x concurrency prompts (8 minimum). One aggregated server, so later prompts'
prefills pause decode: TPOT includes that interference and ITL p90 is the pure decode step.

| Concurrency | Requests | TTFT mean / p90 (ms) | TPOT mean / p90 (ms) | ITL p90 (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 336 / 341 | 10.1 / 10.1 | 10.2 | 96 | 862 |
| 8 | 32 | 1,844 / 2,521 | 13.7 / 14.6 | 13.1 | 517 | 4,655 |
| 32 | 128 | 5,330 / 8,839 | 23.2 / 26.7 | 19.5 | 1,127 | 10,139 |
| 64 | 256 | 9,893 / 17,281 | 33.3 / 40.3 | 25.2 | 1,490 | 13,409 |

The brief's only reference is B300 1P1D decode at 8K context and batch 64: 2,583 tok/s, TPOT p90
26 ms. Steady decode here at 64 running requests is a median 2,557 tok/s (p90 2,582) per the
scheduler log, with ITL p90 25.2 ms, so the GB200 TP16 · DCP16 decode step matches it; the
lower end-to-end output rate comes from serving prefill on the same server. At concurrency 1,
TPOT is 10.1 ms against B300's 9.6 ms. All 424 requests succeeded; no errors in any log.

## Next

- Stage 2 (DFLASH) needs a stop and a start. Per the brief: 8 draft tokens with
  `--max-running-requests` 32 for up to about 8 concurrent requests, 6 for 24-56. PR #48178 is
  not in the branch; its routing is covered by the K3.1 DCP override (see Engine source), but a
  DFLASH + DCP graph-capture stall is the main startup risk.
- Revert `nccl_debug` to WARN at that restart; raise the context to 1,048,576 once stable.
