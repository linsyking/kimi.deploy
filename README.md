# Kimi K3.1 Pro on c04, c11, c13, c14

One SGLang server for moonshotai/Kimi-K3.1-Pro spanning four GB200 trays (16 GPUs) in the
s04-p1-dgx-02 NVL72 rack: TP16 · DCP16, with c04 (10.78.202.32) as rank 0. The API is
OpenAI-compatible on http://10.78.202.32:30000/v1, bound to c04's bond0 address only and
protected by the key in `.secrets/api-key`. Tunable values are in `serving.json`.

- Checkpoint: moonshotai/Kimi-K3.1-Pro @ d9e99b0792a5e93b78ea9bea62a68167e36ddd9e (1.561 TB)
- Draft (DFLASH, stage 2): RadixArk/Kimi-K3.1-Dflash2-Preview @ ea257b38a8ee755b65287bb8ccd0938b519ce26b
- Serving: DFLASH x8, 32 running requests, radix prefix cache on (stage 3), no HiCache
- Engine: SGLang `kimi-k3.1-day-0` @ 8b180ca9e07e, from source in a uv venv (no containers)
- Weights: official shards plus pqhaz's abliteration overlay, active since 2026-11-11 (see Weights)

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
    ./kimictl tunnel start|stop|status   # frp publication on remote port 8313 (see below)
    ./kimictl init-key      # create the API and admin keys (or add a missing admin key)
    ./kimictl weights       # official or abliterated, from every rank's load record
    ./modelctl official|abliterated   # switch weights with rollback (see Weights)
    ./smoke_test.py         # functional checks: health, models, effort tiers, tools, image, 128K needle
    ./anthropic_check.py    # /v1/messages: vision, tools, streaming, effort, web search, thinking budget
    ./claude_code_check.py  # the Claude Code CLI through a recording relay: limits, WebSearch, thinking
    ./bench.sh <label>      # bench_serving 8K in / 1K out at concurrency 1, 8, 32, 64
    ./prefix_bench.py <label>  # multi-turn prefix reuse: TTFT per turn and cache counters

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
enables stage-2 speculative decoding. `radix_cache: true` (stage 3) keeps prompt prefixes on the
GPUs; with `--mamba-max-states-per-path 1` each cached conversation holds one KDA state, and
SGLang sizes 221 state slots, which leaves 8.04M KV tokens (14.3M without the cache). HiCache, the
host-RAM tier, was tried and dropped (DEPLOYMENT_STATUS.md). Both benchmark scripts flush the
prefix cache before measuring.

## API

Use model `kimi-k3.1-pro` with `Authorization: Bearer $(cat .secrets/api-key)`. SGLang's admin
endpoints (`/flush_cache`, weight updates, memory release, pause, profiling) take only the admin
key in `.secrets/admin-api-key`, which is not shared; `/health` and `/metrics` need no key. At the
user's request the key and the endpoint URL (nothing else) are also published, readable by
every user on c04, in `/scratch_local/user_data/kimi-k3.1-pro-endpoint.txt`. After `init-key` and a
restart, rewrite that file too, or it will hand out a dead key. Sampling is
fixed by the model (temperature 1.0, top_p 0.95); don't override it. `reasoning_effort` takes
`low`, `high` or `max` (default `max`); `minimal` maps to `low`, `medium` to `high` and `xhigh`
to `max`. The effort is a prompt instruction to the model, not a token limit. The engine branch
passes `medium` through as a K3.1 tier that this checkpoint's encoder (`encoding_k3.py`) rejects
(HTTP 500), so `serving.json`'s `medium_effort_as: "high"` maps it in the API process, as SGLang
does for K3. `max_tokens` (or `max_completion_tokens`) caps reasoning and answer together, exactly;
a small cap can end inside the reasoning with no answer. The context limit of 262,144 tokens
includes `max_tokens`. Reasoning and tool calls use the `kimi_k3` parsers; in
multi-turn use send back the full assistant message, including `reasoning_content` and
`tool_calls`. Images are accepted as input.

## Anthropic API and Claude Code

SGLang serves the Anthropic Messages API at `http://10.78.202.32:30000/v1/messages` (through frp,
`http://127.0.0.1:8313` on the frp server). For Claude Code set `ANTHROPIC_BASE_URL` to that base
URL, `ANTHROPIC_AUTH_TOKEN` to the API key (Bearer auth; SGLang does not read `x-api-key`, so not
`ANTHROPIC_API_KEY`) and the model to `kimi-k3.1-pro`. Claude Code 2.1.322 sends adaptive thinking
with `output_config.effort` (`high` by default); `CLAUDE_CODE_MAX_OUTPUT_TOKENS` becomes
`max_tokens`, and `MAX_THINKING_TOKENS` sends no numeric budget for this model. Sessions longer
than the 262,144-token context minus `max_tokens` get HTTP 400.

Web search (`enable_web_search`): `web_search.py`, ported from ds.deploy, is an ASGI middleware on
rank 0. A `/v1/messages` request that declares Anthropic's server tool (`web_search` or
`web_search_YYYYMMDD`, as Claude Code's WebSearch sends) gets one Exa search (auto, 5 results) on
the last user text or Claude Code's "for the query:" text; the excerpts are appended to that user
turn as evidence and the server tool is removed before inference. Identical queries are cached for
ten minutes; a failed search returns HTTP 502 rather than an unsearched answer. `allowed_domains`
or `blocked_domains` become Exa filters (not both). No Anthropic server-tool events, citation
blocks or multi-search loops. Requests without the server tool pass through byte for byte. The
Exa key is `.secrets/exa-api-key` (mode 0600, the same key ds.deploy uses); rank 0 reads it at
launch, it is not put in the environment, synced or logged.

Thinking budget (`enforce_thinking_budget`): `thinking_budget.py`, ported from ds.deploy, maps an
Anthropic `thinking.budget_tokens` (1024 or more) onto SGLang's strict-thinking grammar, which
closes the reasoning channel when the budget is spent and lets the answer follow; OpenAI clients
can pass `custom_params: {"thinking_budget": N}`. kimictl adds `--enable-strict-thinking` on every
rank and loads the patch in every process (`KIMI_THINKING_BUDGET=1`). Requests without a budget
or a structured-output constraint skip the grammar, so they keep the fast path. Effort and budget
are independent: effort steers how much the model wants to think, the budget is a hard cap.

## Public access (frp)

`kimictl tunnel start` publishes the API through frp as the single rule `kimi-c04-30000-to-8313`:
ssh.linsyking.com's loopback port 8313 forwards to 10.78.202.32:30000, so on that server the API is
http://127.0.0.1:8313/v1. frpc (config `.secrets/frpc.toml`, mode 0600, which holds the frp token)
dials the frp server through a local gost HTTP proxy on 127.0.0.1:18080 that resolves names over
HTTPS; `tunnel start` starts gost if nothing listens there. It refuses unless the API is healthy,
answers 401 without the key and keeps its admin endpoints from the API key, and it waits for
`[kimi-c04-30000-to-8313] start proxy success` in `.runs/frpc.log`. `kimictl stop` closes the tunnel
first; nothing restarts it. c02's DeepSeek tunnel (remote 8312) is separate and never touched.

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

### Official or abliterated weights

    ./modelctl abliterated   # restart with pqhaz's refusal-reduction patch
    ./modelctl official      # restart with the original weights
    ./modelctl status        # which weights all 16 ranks actually loaded (kimictl weights)

The patch is pqhaz/Kimi-K3.1-Pro-abliteration-patch-20261105 @ f9c1b234cd8ab798a1d042e59de8bf7de938f344.
It rewrites 278 BF16 tensors in 93 of the 96 shards: attention `o_proj`, the dense and shared-expert
`down_proj`, and `routed_expert_up_proj`. Its own script makes a patched copy of the checkpoint;
this deployment does what ds.deploy does for its Heretic weights instead. Both variants read the
same official shards, and nothing is copied, patched on disk or removed. The only extra weight file
is a 29.7 GB `overlay.safetensors` holding the 278 patched tensors, on every node under
`.runs/abliteration/f9c1b234cd8ab798a1d042e59de8bf7de938f344/`. At load, every rank's
`DefaultModelLoader.load_weights_only` swaps them in by name, before TP sharding. This is a runtime
patch from `checkpoint_overlay.py`; the installed SGLang files are untouched. The quantized
experts, vision, tokenizer, encoder and DFLASH draft load unchanged.

Making the overlay, on c04 (each step checks the previous one):

    ./checkpoint_overlay.py download                      # 278 compressed XOR deltas, 14.7 GB
    .venvs/sglang/bin/python -I checkpoint_overlay.py build   # deltas + official shards -> overlay
    ./checkpoint_overlay.py distribute                    # rsync to c11, c13, c14; verify on each

`build` checks every step against the patch manifest:

- the manifest itself, pinned by SHA-256 in `checkpoint_overlay.py`;
- each delta file;
- each original tensor, before the XOR;
- each patched tensor.

`verify` hashes the overlay on a node and writes `overlay.verified.json` for that exact file. `kimictl` preflight then runs a cheap check on every node:

- the pinned manifest;
- where each tensor sits in the official checkpoint;
- the overlay header and size;
- the verified marker.

Every rank records its overlay load in `.runs/abliteration/loads/`. These steps bind their memory to Grace RAM
(`numactl --membind`) and drop the pages they cached, so they can run beside a live server.

`modelctl` checks the overlay on all four nodes before it stops anything. Only `checkpoint_overlay`
in serving.json changes. If the start fails, or fewer than 16 ranks record a full overlay load,
`modelctl` restores the previous serving.json and service; it never forces a stop. A failed
overlay check inside a rank is recorded, not raised, for the same reason: a rank that raises
during load leaves its peers to a forced stop. The model name, keys and other settings stay the
same. `kimictl stop` also stops the tunnel, so start it again after checking the server.

pqhaz reports a partial refusal reduction: 86/89 refusals before, 37/89 after. Benign completions
(58/60) and HumanEval (30/30) were unchanged. Those figures come from vLLM on ROCm at 4K context
and low effort, not from this SGLang build. The DFLASH draft was trained on the official model,
so its acceptance rate may fall.
