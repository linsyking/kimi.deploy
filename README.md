# Kimi K3.1 Pro on c04, c11, c13, c14

One SGLang server for moonshotai/Kimi-K3.1-Pro spanning four GB200 trays (16 GPUs) in the
s04-p1-dgx-02 NVL72 rack: TP16 · DCP16, with c04 (10.78.202.32) as rank 0. The API is
OpenAI-compatible on http://10.78.202.32:30000/v1, bound to c04's bond0 address only and
protected by the key in `.secrets/api-key`. Tunable values are in `serving.json`.

- Checkpoint: moonshotai/Kimi-K3.1-Pro @ d9e99b0792a5e93b78ea9bea62a68167e36ddd9e (1.561 TB)
- Draft (DFLASH, stage 2): RadixArk/Kimi-K3.1-Dflash2-Preview @ ea257b38a8ee755b65287bb8ccd0938b519ce26b
- Engine: SGLang `kimi-k3.1-day-0` @ 8b180ca9e07e, from source in a uv venv (no containers)

Current state and measurements are in DEPLOYMENT_STATUS.md.

## Operations

Run these on c04, in this directory:

    ./kimictl preflight     # GPUs idle, recovery None, fabric healthy, IMEX, files, ports
    ./kimictl start         # sync runtime files, preflight, start all four ranks, wait for /health
    ./kimictl status
    ./kimictl logs [--node c11] [-n 200] [-f]
    ./kimictl restart       # stop, then start only if every GPU came back free
    ./kimictl stop [--force] [--drain-timeout 600]
    ./kimictl sync          # push kimictl, serving.json and docs to the workers
    ./smoke_test.py         # functional checks: health, models, effort tiers, tools, image, 128K needle
    ./bench.sh <label>      # bench_serving 8K in / 1K out at concurrency 1, 8, 32, 64

Each node runs one rank in a tmux session (`kimi-c04` ... `kimi-c14`) on the node-local
socket `.runs/kimi-tmux`, logging to `sglang.log` in this directory on that node; earlier
logs move to `logs/`. `start` refuses to run if any GPU has compute processes or memory in
use, a recovery action other than None, or an unhealthy fabric, or if a rank is already up.
If a rank exits during startup, `start` prints every node's log tail (read c04's first: a
rank that dies takes the others down with gloo "connection closed" errors) and stops the
rest. Startup loads 1.5 TB and captures CUDA graphs: about 11 minutes (weights 265 s, decode
graphs 250 s). On a readiness timeout the ranks are left running for inspection. Nothing here
resets GPUs, reboots nodes, or restarts a crashed service automatically. There is no root on
these nodes, so GPU memory that a dead process failed to release cannot be recovered here.

`stop` goes through SGLang's own shutdown. It sends SIGTERM to rank 0's main process only,
which finishes in-flight requests, then broadcasts ShutdownReq so every rank's schedulers
free their GPU state in userspace and exit; the followers then exit by themselves. Each node
waits until nvidia-smi shows every GPU free again and reports the memory left. If a rank
cannot drain (a dead peer or a hang), `stop` lists the remaining PIDs and does nothing
more; `stop --force` then sends SIGTERM and finally SIGKILL to those ranks' process groups.
SIGKILL mid-release is how GPU memory outlives a process, so use `--force` only after
reading the logs. When a rank dies during startup, `start` forces the stop itself, because
the surviving ranks are blocked on the dead one.

Ranks run under `numactl --membind=0,1`. On GB200 each GPU's HBM is also a CPU-less NUMA
node (2, 10, 18, 26) and Grace RAM (958 GiB) is smaller than the checkpoint, so page cache
from reading the weights would otherwise spill into HBM. nvidia-smi counts that page cache
as GPU memory in use, and it shrinks the KV pool unevenly across ranks. The policy affects
host memory only; CUDA allocations still go to the GPU. `start` first drops the model files'
cached pages (`evict_page_cache.py`, posix_fadvise; no root), and preflight refuses to start
while 1 GiB or more of page cache sits in HBM.

Edit `serving.json` and restart to change engine settings. `extra_args` and `extra_env`
append raw SGLang flags and environment variables for experiments. Setting `speculative` to
`{"algorithm": "DFLASH", "draft_model_path": "/scratch_local/user_data/yiming/models/Kimi-K3.1-Dflash2-Preview", "num_draft_tokens": 8}`
enables stage-2 speculative decoding.

## API

Use model `kimi-k3.1-pro` with `Authorization: Bearer $(cat .secrets/api-key)`. At the
user's request the key and the endpoint URL (nothing else) are also published, readable by
every user on c04, in `/scratch_local/user_data/kimi-k3.1-pro-endpoint.txt`. After `init-key` and a
restart, rewrite that file too, or it will hand out a dead key. Sampling is
fixed by the model (temperature 1.0, top_p 0.95); don't override it. `reasoning_effort` takes
`low`, `high` or `max` (default `max`); `minimal` maps to `low` and `xhigh` to `max`. Don't
send `medium`: the engine branch passes it through as a K3.1 tier, but this checkpoint's
encoder (`encoding_k3.py`) accepts only the three above, so the request fails with HTTP 500.
Reasoning and tool calls use the `kimi_k3` parsers; in
multi-turn use send back the full assistant message, including `reasoning_content` and
`tool_calls`. Images are accepted as input.

## Environment

`build_venv.sh` builds `.venvs/sglang` on c04 from the pinned checkout in `.src/sglang`,
reproducing the Python layer of the branch's `docker/k31.Dockerfile` with uv:

- CPython 3.12.13 (uv-managed), torch 2.14.1+cu130, the pyproject pins locked in
  `requirements-sglang.lock`, NCCL 2.30.7, FlashInfer 0.7.3 with its cubin and cu130 jit-cache.
- The CUDA toolchain comes from pip wheels: `CUDA_HOME=<site-packages>/nvidia/cu13`, kept on
  the 13.0 series by `constraints-cuda130.txt` to match torch, the image and driver 580.
- SGLang is installed non-editable with its bundled Rust extensions (TreeCore and others),
  built with Rust 1.92 from `~/.local/opt/rust` (c04 only), so the venv is self-contained and
  the workers need no Cargo (`SGLANG_RUST_BUILD_MODE=never` at launch).
- The K3.1 FlashInfer DynBlock step raises the routing kernel's expert limit to 1024 (the
  model has 896) and rebuilds the jit-cache modules that include it. Without it SGLang still
  runs, only slower. The image's Prims-TS overlay (opt-in backend) and Dynamo are not used.

`verify_venv.py [--gpu]` checks the pins, the K3.1 model class, TreeCore, DynBlock and the
toolchain on a node. `replicate_venv.sh` rsyncs the venv (and the uv Python if missing) to the
same path on c11, c13 and c14 and runs the check there. Kernel caches are node-local under
`.runs/kernel-cache/sglang`; the launcher sets `HOME` to the scratch home because
non-interactive SSH gets the NFS home.

Tools on every node: `~/.local/bin/{rsync,tmux,uv}` (c14's were copied from c04's conda-forge
prefix `~/.local/opt/sysutils`). No node has a system rsync, and `~/.local/bin` is not on the
non-interactive SSH PATH, so scripts use absolute paths.

## Weights

`stage_weights.sh download|distribute|verify pro|draft` downloads a snapshot once on c04
into a flat directory under `/scratch_local/user_data/yiming/models` (not the HF cache; its
symlinks break `trust_remote_code`), rsyncs it to the workers over bond0 in parallel streams,
and verifies every file on every node against `manifests/*.manifest.json` (HF LFS sha256 and
git blob ids). `verify_weights.py` writes `.kimi-verified.json` into the model directory;
`kimictl` refuses to start without it. The service runs with `HF_HUB_OFFLINE=1`.
