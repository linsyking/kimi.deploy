# Deployment status — 2026-11-10

Kimi K3.1 Pro, TP16 · DCP16 across c04 (rank 0), c11, c13 and c14, on
http://10.78.202.32:30000/v1 (model `kimi-k3.1-pro`). Stage 3 (DFLASH, 8 draft tokens, radix
prefix cache without HiCache, 262,144 context) plus the Anthropic web search and thinking budget
ported from ds.deploy has served since 04:36 UTC on 2026-11-11 (20:36 PST) and is published through
frp on remote port 8313. Every item of the brief's validation checklist passes for all stages.

| Stage | State |
|---|---|
| 0. Pre-flight | passed |
| Environment (venv) | built on c04, replicated and verified on all four nodes |
| Weights | downloaded on c04, rsynced, verified on all four nodes |
| 1. Baseline launch | serving; functional checks, stress test and bench sweep pass |
| 2. DFLASH | serving; functional checks, stress test and bench sweeps pass |
| 3. Prefix cache | serving with the radix cache; HiCache tried and dropped at the user's decision |
| Anthropic extras | web search (Exa), thinking budget and a medium-effort mapping; all checks pass |
| Abliteration overlay | serving since 08:36 UTC on 2026-11-11: pqhaz's patch as a 278-tensor overlay, loaded on all 16 ranks |

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
network. The API key was generated with `kimictl init-key` into `.secrets/` (mode 0600;
ds.deploy's secrets are not reused).

With only `--api-key`, SGLang accepts that key on its admin endpoints too (weight updates, memory
release, pause, profiling, `/flush_cache`), and the key is shared and published through frp. Since
the 20:32 UTC restart a separate admin key (`.secrets/admin-api-key`, mode 0600, not shared) is
set through the same auth config; the admin endpoints then accept only it. Checked: the API key
gets 401 on an admin endpoint, the admin key passes auth there and gets 401 on `/v1/models`.
`/health` and `/metrics` answer without a key (SGLang always allows them).

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

## Stop between stages

`kimictl stop` at 19:44 UTC, idle server: rank 0 took SIGTERM, drained 0 requests and broadcast
ShutdownReq; every scheduler on all four nodes exited with status 0 within 2 s (no "still alive
... killing" warning), and each node saw its GPUs back at 2-6 MiB with no compute processes and
recovery action None, 8.5 s in all. No processes were left. Rank 0 left three files in /dev/shm
(two SGLang message queues of 24 MB apparent size and a 16 KB load-stats file, named by PID so
they cannot collide). The queues are sparse: each file holds one 64 KB page, so a run leaves
192 KB of host RAM, not 48 MB. Every run of rank 0 leaves its set; the workers leave none. They are
outside `kimi.deploy/`, so they were left for the user to decide on. Preflight then flagged ports 30000 and 20000 as in use: only TIME-WAIT sockets from the
stop. Its bind test now sets SO_REUSEADDR like uvicorn and torch's TCPStore.

## Stage 2: DFLASH

`serving.json`: `speculative` = DFLASH with the RadixArk/Kimi-K3.1-Dflash2-Preview draft and 8
draft tokens (`--speculative-draft-attention-backend cutedsl_mla --speculative-draft-kv-cache-dtype
fp8_e4m3 --speculative-use-block-verification --enable-linear-replayssm-spec`), and
`--max-running-requests` 32, the brief's setting for width 8 (the draft card lowers it to 32 at
width 8 because the speculative KDA state cache grows with width x requests). `NCCL_DEBUG` back to
WARN.

Launch (19:45 UTC): ready after 422 s. Target weights 263 s, draft weights 0.6 s, target-verify
CUDA graphs for batch 1-32 in 39 s, draft graphs in 24 s; no capture stall (see Engine source for
why PR #48178 does not apply). SGLang logged the expected `DFLASH block size mismatch: using
speculative_num_draft_tokens=8 but draft config block_size=5`, the DFLASH draft KV ring (66 rows
per request per rank), KDA verify on `nv_cutedsl` and DSA attention on `flashmla_kv`. KV pool:
14,323,712 tokens (15.77 GB per GPU); KDA state pool: 32 slots.

Functional checks all pass again (`.runs/smoke/20261110-115500.json`): health, models, effort low
and max, tool call round trip, image, and the needle at 127,902 tokens (5.2 s). The 32 x 8K/1K
stress test right after startup finished in 18.5 s (30.1 s in stage 1), all 32 requests, no stall.

Benchmarks. Acceptance length is the run's generated tokens per verify step, from `/metrics`
counter deltas (the server's own average is cumulative since startup).

Fixed 8K input / 1K output (`bench.sh stage2`), comparable with stage 1:

| Concurrency | Requests | TTFT mean / p90 (ms) | TPOT mean / p90 (ms) | ITL p90 (ms) | Output tok/s | Total tok/s | Accept length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 343 / 349 | 1.9 / 2.1 | 2.0 | 446 | 4,013 | 7.39 |
| 8 | 32 | 1,244 / 1,918 | 4.6 / 6.5 | 4.3 | 1,233 | 11,097 | 7.26 |
| 32 | 128 | 4,195 / 7,420 | 10.7 / 13.5 | 5.4 | 2,064 | 18,574 | 7.59 |
| 64 | 256 | 17,297 / 21,735 | 11.5 / 13.6 | 6.4 | 2,066 | 18,598 | 7.60 |

These overstate DFLASH: the random dataset repeats a short ShareGPT prompt up to 8K tokens, the
output turns repetitive, and the draft is accepted almost every step (7.3-7.6 of 8). At
concurrency 64, half the requests queue behind the 32-request cap, so throughput stays at the
c32 level.

Natural text (`DATASET=sharegpt OUTPUT_LEN=2048 bench.sh stage2-sharegpt 1 8 32`): real ShareGPT
first turns as chat requests, reasoning at the model's default (max), stopping on their own up to
2,048 tokens:

| Concurrency | Requests | TTFT mean / p90 (ms) | TPOT mean / p90 (ms) | ITL p90 (ms) | Output tok/s | Total tok/s | Accept length |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 262 / 269 | 4.5 / 5.6 | 13.3 | 215 | 244 | 2.74 |
| 8 | 32 | 348 / 508 | 7.7 / 10.8 | 24.1 | 921 | 1,127 | 2.89 |
| 32 | 128 | 1,984 / 3,555 | 12.0 / 17.8 | 39.9 | 2,217 | 2,587 | 2.92 |

Acceptance on mixed chat is 2.7-2.9, below the draft card's 3.4-5.8 at width 8 (GSM8K, AIME26,
GPQA). At concurrency 1, TPOT is 4.5 ms against stage 1's 10.1 ms, about 2.2x faster per user;
ITL counts per chunk, and each verify step emits about three tokens. Stage 1 was not run on this
dataset, so the c8/c32 comparison is against its 8K/1K numbers only.

## Stage 3: prefix cache

### Trial with HiCache (20:11-20:31 UTC)

`serving.json`: `radix_cache` on, plus the brief's HiCache flags (`--enable-hierarchical-cache
--hicache-size 120 --hicache-io-backend direct --hicache-mem-layout page_first_direct
--mamba-max-states-per-path 1`), with stage 2's DFLASH x8 and 32 running requests. Ready after
381 s (`HiCache + DCP enabled (dcp_size=16)`).

- GPU: the radix cache needs a KDA state per cached prefix, so the state pool grew from 32 slots
  to 222 (5.63 GB per GPU) and the KV pool shrank from 14,323,712 tokens to 8,091,648 (505,728 per
  rank, 8.91 GB).
- Host, per rank: 3,972,096 KV tokens (62.54 GB; 63.55M logical tokens across DCP) and 2,047 KDA
  states (57.49 GB). That is about 247 GiB of mlocked memory on each of Grace NUMA nodes 0 and 1,
  per node (958 GiB in all), held by the processes and not as /dev/shm files, so the kernel frees
  it at exit.
- Warnings: KDA state copies use a synchronous torch fallback, and the `direct` backend's
  copy-engine DMA "can queue ahead of the forward's small metadata host-to-device copies and stall
  the forward" (the engine suggests `SGLANG_ENABLE_MLP_SYNC_METADATA_FILL=1`; not tried).

Functional checks all passed (`.runs/smoke/20261110-121914.json`). The 32 x 8K/1K stress test
right after startup completed 32/32 in 24.0 s (stage 2: 18.5 s): TTFT mean 11.3 s (5.5 s), TPOT
mean 9.7 ms (9.8 ms), worst ITL 7.2 s (3.5 s). The first 8K prefill after startup took 7 s against
3 s in stage 2; steady prefill was about 27K tok/s in both. The rest was not isolated.

Prefix reuse (`prefix_bench.py`): 96 sessions, each with its own ~24K-token document, 3 turns
sent turn-major (every session's turn 1, then every turn 2, ...), concurrency 8, `reasoning_effort`
low and 256 output tokens. Turn k resends the whole conversation.

| Turn | Stage 2, no cache: TTFT mean / p50 / p90 (s) | Stage 3 with HiCache: TTFT mean / p50 / p90 (s) |
|---:|---:|---:|
| 1 | 1.52 / 0.98 / 1.86 | 1.56 / 1.03 / 1.89 |
| 2 | 2.25 / 1.88 / 3.51 | 0.60 / 0.65 / 0.94 |
| 3 | 1.84 / 1.71 / 3.22 | 0.57 / 0.64 / 0.78 |
| Run time | 306 s | 180 s |

All 4,521,984 cached prompt tokens came from GPU memory (`cached_tokens_total{cache_source=
"device"}`); HiCache copied 2.72M tokens (59 GB) per rank to host and loaded none back. The 96
sessions (2.3M tokens, 96 KDA states) fit in GPU memory, so the gain is the radix cache's alone.
The host tier only matters past about 8.1M cached tokens or about 200 cached conversations (one KDA
state each). A larger test (320 sessions x 30K tokens) was stopped in its first turn at the user's
request, after the host tier had served 400,384 prompt tokens; it has no latency figures, and its
first 96 sessions partly hit the previous run's documents (same seeds; `prefix_bench.py` now
flushes the cache first).

Decision (user, 20:30 UTC): keep the radix cache, drop HiCache. It added host copy traffic, a
possible forward stall and 494 GiB of locked RAM per node, and helps only for more live sessions
than this deployment expects.

Stop at 20:31 UTC: rank 0 drained 0 requests; every GPU came back at 4-9 MiB with no processes,
and locked host memory fell to under 200 MB on c04 and 0 on the workers.

### Radix cache only

`serving.json`: `radix_cache` on with `--mamba-max-states-per-path 1` (each cached conversation
keeps one KDA state: the tail), no HiCache flags. Restart at 20:32 UTC (`kimictl restart`): ready
after 371 s. KDA state pool 221 slots (5.61 GB per GPU), KV pool 8,037,376 tokens (502,336 per
rank, 8.85 GB), `hicache_attached=False`. No host pools; locked host memory stays under 200 MB.

- Stress test right after startup (`.runs/bench/stage3-radix-stress-c32.log`): 32/32 in 24.0 s,
  TTFT mean 11.5 s, TPOT mean 9.7 ms / p90 13.4 ms, worst ITL 3.5 s. TPOT and worst ITL match
  stage 2 (9.8 ms, 3.5 s); the HiCache trial's 7.2 s worst ITL is gone. TTFT is still twice stage
  2's 5.5 s: the first 8K prefill after startup took 8 s against 3 s, and the other 31 queued
  behind it. It is a one-off cost of the first request with the radix cache on; later prefills
  ran at stage 2's rate.
- Functional checks all pass (`.runs/smoke/20261110-123913.json`), needle at 127,902 tokens in
  5.3 s.

The prefix-reuse gain measured in the trial came from this same device cache. No further
benchmarks were run, at the user's request.

## Anthropic extras: web search, thinking budget, effort (2026-11-10, 20:29 PST restart)

The brief's new section asks for ds.deploy's Anthropic web search and thinking budget. Both are
runtime patches; the SGLang package is unchanged.

- `web_search.py` (from ds.deploy, docstrings only changed): installed by `kimictl _engine` on rank 0
  before `run_server`. This build has `/v1/messages` and the `ExaClient` members it uses
  (`_search_payload`, `_post`, `close`, `ExaSearchConfig(num_results, search_type,
  include_highlights)`); `test_web_search.py` passes its 14 tests plus a new one that drives the
  installed client without network. Exa is reachable from c04 directly (no proxy). The key is read
  from `.secrets/exa-api-key` on rank 0 only; unlike ds41ctl it is not put in the engine
  environment, so the workers never see it.
- `thinking_budget.py` (from ds.deploy, marker renamed): loaded at import of `kimictl` when
  `KIMI_THINKING_BUDGET=1`, so in the API process and every scheduler on all four nodes;
  `--enable-strict-thinking` on every rank and `SGLANG_MAX_THINK_TOKENS=262144` as the unreachable
  default. In this build strict thinking attaches a grammar to every unconstrained request; the
  patch keeps requests without a budget or a structured-output constraint off it.
  `test_thinking_budget.py` checks the adapter offline. The engine was ready for it: the
  `kimi_k3` reasoning detector defines its strict-thinking exclusions, XGrammar already initialized
  with this tokenizer, and DFLASH v2 applies grammar masks to its draft chain.
- `medium_effort_as: "high"`: Anthropic `output_config.effort="medium"` (which Claude Code sends
  when its effort is set to medium) and OpenAI `reasoning_effort="medium"` returned HTTP 500,
  because the branch passes `medium` through as a native K3.1 tier and this checkpoint's encoder
  accepts only low, high and max. kimictl maps it to `high` in the API process, the mapping SGLang
  uses for K3.

Launch: ready after 351 s, `Anthropic Exa search adapter enabled`, `enable_strict_thinking: True`.
The 32 x 8K/1K stress test passed (32/32 in 24.7 s, TPOT mean 9.7 ms, worst ITL 2.4 s), and the
functional checks pass (`.runs/smoke/20261110-203709.json`).

`anthropic_check.py` (`.runs/checks/anthropic-20261110-20*.json`), all passing:

| Check | Result |
|---|---|
| Vision, JSON and stream | "left: red, right: blue" |
| Tool call and tool result, JSON and stream | `fixture_lookup({"key": "deployment"})`, then the returned value |
| Effort low / medium / high / xhigh / max, both APIs | all 200 (medium was 500 before the mapping) |
| Web search, `web_search_20250305` JSON and `web_search_20260209` stream, allowed_domains | one Exa search (the second hit the cache), answers cite github.com/sgl-project/sglang |
| Client tool named `web_search` (not the server tool) | passed through; no Exa call |
| Budget 16 and 128 via OpenAI `custom_params`, raw token ids | the close marker `<\|close\|>think<\|sep\|>` starts at exactly token 16 and 128; an answer follows (with 16 the model kept reasoning in the answer until `max_tokens`) |
| Anthropic `budget_tokens` 1024 and 2048, JSON and stream, all four in one batch | reasoning re-tokenizes to exactly 1024 and 2048 tokens, then an answer, `end_turn` |
| Budgeted tool call and result | pass |

Unbudgeted decode, one stream at a time (8 requests before the restart, 16 after): mean 200 tok/s
before and 187 after (medians 202 and 184). The difference is about 1.5 standard errors at
temperature 1.0 with DFLASH acceptance varying per request, so it is within noise as measured;
by construction those requests take no grammar.

`max_tokens` is enforced exactly, checked before the restart (`.runs/checks/max-tokens-pre.json`):
64-token caps stopped at 64 on OpenAI chat (`finish_reason=length`, streaming too, and
`max_completion_tokens`) and on `/v1/messages` (`stop_reason=max_tokens`, streaming too, thinking on
or off). Reasoning counts toward it; DFLASH never overshoots.

Claude Code 2.1.322 (`claude_code_check.py`, `.runs/claude-code/20261110-204014/`): `claude -p`
with `ANTHROPIC_BASE_URL` at a loopback relay that records only limit and tool fields, Bearer
auth, and a fresh HOME and config directory. All four pass:

- basic: one request, answer correct.
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS=256`: four requests, each `max_tokens=256`, each stopped at 256
  with `max_tokens`; Claude Code then reports "response exceeded the 256 output token maximum".
  Its default is `max_tokens=32000`.
- WebSearch: the main turn calls the tool, Claude Code sends one `web_search_20250305`
  subrequest, the adapter answers it from Exa, and the final reply is the SGLang GitHub URL.
- `MAX_THINKING_TOKENS=2048`: Claude Code sends `thinking: {"type": "adaptive", "display":
  "omitted"}` and `output_config.effort: "high"`, no numeric budget, as ds.deploy found for its
  model. Budgets therefore apply to clients that send `budget_tokens` (or `custom_params`), not
  to Claude Code.

Traffic seen on the previous run (12:38-20:29 PST): 141 `/v1/messages` and 13 chat completions
succeeded. Three `/v1/messages` requests from one Claude Code session got HTTP 400 at 18:47: 232,161
input tokens plus `max_tokens` 32,000 exceed the 262,144-token context.

## Abliteration overlay (prepared 2026-11-11 07:45-08:00 UTC; serving since 08:36 UTC)

Searching Hugging Face on 2026-11-11 found these uncensored K3.1 Pro variants; the last column is each card's own
quality report. Sizes are the HF API's per-file sizes (an earlier, truncated listing gave 514–644 GB for the gated repos).

| Repo | What it is | Size and access | Quality reported on other prompts |
|---|---|---|---|
| pqhaz/Kimi-K3.1-Pro-abliteration-patch-20261105 | exact XOR patch of 278 BF16 tensors; refusals 86/89 → 37/89 | 14.7 GB; not gated | small samples, raw data published:<br>• benign prompts: 58/60 answered, before and after<br>• HumanEval: 30/30 passed, before and after<br>Low effort, 4K context, vLLM on ROCm, one judge model |
| lyber-ai/Kimi-K3.1-Pro-UNCENSORED-MXFP4 (ABLITERATED-MXFP4 is an alias) | full checkpoint; HarmBench 320/320 complied | 1,566 GB; manual approval | full MMLU, 5-shot, logit mode, no reasoning:<br>• all subjects: 90.97% → 85.66% (−5.3 pp)<br>• without moral_scenarios: −3.0 pp<br>• college physics −11.8 pp, econometrics −13.2 pp, professional law −8.7 pp<br>No reasoning-mode, coding or agent tests |
| audnai/penclaw-Kimi-K3.1-pro-unrestricted | full checkpoint, abliteration only; 33.9% of requests on its 519-prompt refusal benchmark answered | 1,561 GB; manual approval, opts you into marketing email | none; the card only calls coherence "intact" |
| Blackfrost-Research/Kimi-K3.1-Pro-DERISKED | refusal-direction edit plus cyber SFT | 1,628 GB listed; the card says the weights are pending and unlocked after purchase | none; the card says no evaluation claims yet |
| routerhub/kimi-k3.1-pro-abliterated | no weights; points to a hosted service | none | none |

Only pqhaz's patch avoids a new full 1.56 TB download. Its quality samples are too small to show a drop of a few points, so "unchanged" there is a weak claim.

- pqhaz's patch, rev f9c1b234cd8ab798a1d042e59de8bf7de938f344, variant `trace4`:
  - It rewrites 278 BF16 tensors in 93 of the 96 shards: 93 `o_proj`, 93 `down_proj` and 92 `routed_expert_up_proj`.
  - It names base revision 6d0a0202. That revision's 96 safetensors and index are byte-identical to ours (d9e99b07); only `encoding_k3.py` differs. The index SHA-256 and every original tensor's SHA-256 match ours.
  - Its `apply_exact_patch.py` copies every shard before patching, which here would mean 1.56 TB per node.
  - We used none of its code.
- Applied as ds.deploy does (README, Weights): one overlay file of the patched tensors, substituted by name in every rank's weight loader.
  - `checkpoint_overlay.py download`: 278 files, 14.70 GB, kept on c04 only under `.runs/abliteration/f9c1b234…/source/`.
  - `build` on c04 took 112 s. Every delta, original and patched tensor matched the manifest.
  - `distribute` to c11, c13 and c14 took 45 s. `verify` hashed all 278 tensors on each node in 2 s.
  - Disk: `overlay.safetensors` is 29,697,811,472 bytes on every node. The official shards are unchanged.
- Spot check: the overlay tensors differ from the official ones in about 90% of entries, by 2–8% in norm (|dW|/|W| = 0.024 to 0.079 for layers 0, 1, 51 and 92). That fits a projection along a few directions.
- All of this ran beside the live server, bound to Grace RAM: rsync on both ends, build and verify all show memory policy `bind:0-1`. HBM page cache stayed at 0–320 kB on every node; the 320 kB were already there before. The service kept serving; `kimictl status` stayed healthy.
- Tests: 16 offline tests in `test_checkpoint_overlay.py` cover:
  - build and verify on a synthetic checkpoint;
  - the load hook on SGLang's real `DefaultModelLoader`, for the target, the draft and a failed check;
  - modelctl rollback.
- Not yet measured on this deployment:
  - refusals, quality, and DFLASH acceptance with the overlay;
  - load time, which should grow a little: each rank also reads the 29.7 GB overlay.
- Switched at the user's request on 2026-11-11 with `./modelctl abliterated`, without benchmarks:
  - The graceful stop drained 0 requests; every GPU came back to 3–8 MiB.
  - Ready after 370 s. Weight loading took 261.6–265.0 s per c04 rank, against 262.5–264.9 s with the official weights, so the overlay adds no measurable load time. Memory use is unchanged: 132.88 GB per GPU for the target, 1.36 GB for the draft, which loaded untouched.
  - `kimictl weights` reports 278 tensors loaded on 16/16 ranks.
  - `smoke_test.py` passes: health, low and max effort, tool call, image, and a 128K-token needle.
  - The node logs show only the 4 NCCL "ibverbs lib wrapper not initialized" warnings per worker that the official run also had.
  - The tunnel reopened at 00:37:39 PST (08:37:39 UTC).
- Not measured: refusals, quality and DFLASH acceptance with the overlay. `./modelctl official` goes back.

## Public access (frp)

`kimictl tunnel start` at 19:55 UTC, once stage 2 passed its checks: config check (exactly the
`kimi-c04-30000-to-8313` rule, targeting 10.78.202.32:30000), API healthy and 401 without the key,
`frpc verify`, gost started on 127.0.0.1:18080 and a DNS-over-HTTPS lookup through it, then frpc:
`login to server success` and `[kimi-c04-30000-to-8313] start proxy success`. frpc and gost each
lead their own session; the frp token appears in no log, state file, doc or commit. Server-local
URL on ssh.linsyking.com: http://127.0.0.1:8313/v1, model `kimi-k3.1-pro`; not tested from the frp
server side. c02's DeepSeek tunnel (remote 8312) was not touched. `kimictl stop` closes the tunnel
before stopping the server; nothing restarts it automatically.

The stop before the stage 3 trial closed it (20:10 UTC). Restarted at 20:39 UTC on the radix-only
server, after its stress and functional checks; `tunnel start` now also refuses unless an admin
endpoint rejects the API key. frpc logged `login to server success` and
`[kimi-c04-30000-to-8313] start proxy success` (12:39:33 PST). No key or frp token appears in any
log, doc, state file or commit.

Closed by the 20:29 PST restart and reopened at 20:40:59 PST after the checks above:
`login to server success`, `[kimi-c04-30000-to-8313] start proxy success`.

At the user's request the API key and the endpoint URL, and nothing else (no frp details), are
in `/scratch_local/user_data/kimi-k3.1-pro-endpoint.txt`, mode 0644, readable by every user on c04.

## Next

- The service runs the abliterated overlay; its effect on quality and on DFLASH acceptance is unmeasured (no benchmarks, at the user's request).
- `/dev/shm` on c04 holds rank 0's leftover queue files from three stopped runs (192 KB each run);
  deleting them, now or in `kimictl stop`, waits on the user.
- HiCache is worth another look only for more than about 200 live long conversations or more than
  8M cached tokens; the context can go to 1,048,576 once the service has run stably.
- Claude Code sessions beyond about 230K tokens get HTTP 400 at the 262,144 context; raising
  `context_length` toward the model's 1,048,576 would help them and needs a restart and re-test.
