# Kimi K3.1 Pro on s04-p1-dgx-02: deployment context

Prepared 2026-11-10 from a session on c02. Facts were verified on the nodes that day unless marked as an estimate. Re-check anything that could have changed (GPU occupancy, disk, other users) before acting on it.

## Goal

Serve `moonshotai/Kimi-K3.1-Pro` as one SGLang server spanning 4 GB200 trays (16 GPUs): c04, c11, c13, c14. Expose an OpenAI-compatible API, and follow the conventions of `ds.deploy` (controller script, `serving.json`, README, status doc). Do the work in `/scratch_local/user_data/yiming/kimi.deploy` on c04. It's an empty git repo on branch `main` with no commits and no remote.

## Rules from the user

- **Never use Docker. Don't use enroot.** Run SGLang from a uv-managed Python venv, the way `ds.deploy` runs `.venvs/sglang`. If an upstream recipe ships only an image, read its Dockerfile for the dependency pins and reproduce them in the venv.
- **Use only c04, c11, c13 and c14.** Don't touch other nodes.
  - c02 runs the production DeepSeek-V4.1-Flash service on port 18000, published through an frp tunnel. Don't stop it, reconfigure it, or reuse its port or secrets.
- **Don't modify other users' files.** For example, `/scratch_local/user_data/models/*` and the `*.sqsh` images on c04 belong to jingxinpan. Reading them is fine.
- **Ask the user before:**
  - starting the 1.56 TB weight download
  - exposing the endpoint outside the cluster in any way other than the approved frp rule on remote port 8313 (see "Public access via frp")
  - deleting anything outside `kimi.deploy/` and the directories you create
- **Never reset GPUs or reboot nodes.** Report recovery actions or hung drivers instead.

## Nodes

| Node | IP | Hostname | Rank | Notes |
|---|---|---|---|---|
| c04 | 10.78.202.32 | s04-p1-dgx-02-c04 | 0 (head) | You run here. Use `--dist-init-addr 10.78.202.32:20000`. |
| c11 | 10.78.202.39 | s04-p1-dgx-02-c11 | 1 | |
| c13 | 10.78.202.41 | s04-p1-dgx-02-c13 | 2 | |
| c14 | 10.78.202.42 | s04-p1-dgx-02-c14 | 3 | Scratch home was just created and is empty. It has no `~/.local` (no uv, hf, rsync, fish or claude) and no uv Pythons, so provision it. |

**Per node:**
- 4× NVIDIA GB200, each with 189,471 MiB (about 185 GiB) of HBM.
- 144 CPU cores and about 1,694 GiB of RAM.
- `/scratch_local`: a 15.2 TB local RAID (`/dev/md0`) with 9–13 TB free. **It is not shared between nodes.**
- aarch64. `/etc/os-release` reports Ubuntu 22.04 LTS (glibc 2.36). Kernel `6.14.0-1015-nvidia-64k`.
- Driver 580.167.08, which supports CUDA 13.0. The system CUDA toolkit is only 12.8 (`/usr/local/cuda-12.8`), so use the CUDA 13 toolchain from pip wheels, as `ds.deploy` does.

**Fabric:**
- The whole rack is one NVL72 NVLink domain: cluster UUID `dedfac56-88ba-46f8-9939-7b7dc85c0dc4`, clique 32766.
- All 4 nodes reported fabric HEALTHY, 72/72 NVLinks active, recovery action None, and idle GPUs (0 MiB used).
- `nvidia-imex` is active on all four. c14 exposes only `/dev/nvidia-caps-imex-channels/channel0`; the others expose channel0–3.

**Network:**
- `bond0` (10.78.202.x/25, 400 Gb/s) carries bootstrap and inter-node traffic. Set `GLOO_SOCKET_IFNAME=bond0` and `NCCL_SOCKET_IFNAME=bond0`.
- The RDMA links `enP6s6f1np1` and `enP22s22f1np1` are up on all four. In-rack TP traffic goes over NVLink, so it doesn't need them.

**Access:**
- User `haiyangshi`. Passwordless SSH from c04 to c02, c11, c13 and c14 works with BatchMode.
- Non-interactive SSH gets `HOME=/home/haiyangshi` (shared NFS) and **doesn't** put `~/.local/bin` on PATH.
- Interactive shells set `HOME=/scratch_local/user_data/yiming`, which is local to each node, via `/home/haiyangshi/.bashrc`.
- **Use absolute paths in every remote command.**

**Tools:**
- No node has a system `rsync`. It exists only at `/scratch_local/user_data/yiming/.local/bin/rsync` on c04, c11 and c13. Use `tar | ssh`, or copy rsync to c14 and pass `--rsync-path`.
- uv 0.11.32 is at `/scratch_local/user_data/yiming/.local/bin/uv` on c04, c11 and c13.
- On c04, `hf` (huggingface_hub 1.30.0) is in the same directory.

**Internet and token:**
- Direct HTTPS to huggingface.co works from all nodes, with no proxy needed. The repos aren't gated.
- The HF token is at `/scratch_local/user_data/yiming/.cache/huggingface/token` on c04. It's a fine-grained token for HF user `kingxiang`, and `hf auth whoami` works with it.
- c11 has the same token at `~/.hftoken`. c13 and c14 have none.

**State:**
- The scratch homes on c04, c11 and c13 were just wiped down to `.config`, `.local` and `.claude` (plus `kimi.deploy` on c04). No old venvs, caches or models remain.

**Occupancy (2026-11-10):**
- None of the 4 had other users' processes, or other users' logins in the last 30 days.
- Other users are active on c06, c07, c08, c10, c12, c15, c17 and c18.
- c03 has a hung GPU driver (`nvidia-smi` stuck in D state). c09 is unreachable. c01 and c02 are Kubernetes workers.
- **Re-check before launch** that nobody has started using the 4 nodes: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, `who`, and other-user processes in `ps`.

## Reference material on c04 (read it; don't run it as-is)

- **`/scratch_local/user_data/yiming/reference/ds.deploy`**: the live DeepSeek-V4.1-Flash deployment from c02. It's a git clone plus the current uncommitted working tree, verified identical. `.secrets`, `.venvs`, `.runs` and logs were not copied.
  - `ds41ctl` is the controller: tmux engine session, start/stop/status, refuses to start if GPUs are busy or a recovery action is pending. Its `launch_environment()` function builds the CUDA 13 toolchain and cache environment.
  - Also: `serving.json`, `modelctl`, `README.md`, `DEPLOYMENT_STATUS.md`.
  - Its venv runs Python 3.11 (uv) with: sglang 0.5.23 (PyPI), sglang-kernel 0.4.10, flashinfer-python 0.7.2 plus flashinfer-jit-cache 0.7.2+cu130, sgl-deep-gemm 0.2.3.post2, sgl-flash-mla 1.0.0, sgl-deep-ep 0.1.2.post1, torch 2.14.1, nvidia-nccl-cu13 2.30.7 and cuda-python 13.4.4.
  - It sets `CUDA_HOME=<site-packages>/nvidia/cu13`.
- **`/scratch_local/user_data/yiming/reference/glm.deploy`**: a GLM-5.3 dual-node SGLang deployment. It's useful for multi-node patterns: `glm53ctl`, `sync_worker.sh`, `serve_glm53_sglang*.sh`, `AGENTS.md`, and the source patch `patches/sglang-0.5.19-http-server-middlewares.patch`.
- **`/scratch_local/user_data/yiming/reference/rack-check/check_rack.py`** (docs in `RACK_CHECK.md`): a read-only health checker covering GPUs, NVLink fabric, RAM and login history.
  - It takes a contiguous range, so for these nodes run `--start 4 --end 4`, `--start 11 --end 11` and `--start 13 --end 14`.
  - Avoid c03: each probe there leaves another hung `nvidia-smi`.

## Model facts

- **Repo:** `moonshotai/Kimi-K3.1-Pro`, revision `d9e99b0792a5` (modified 2026-11-05). 112 files, 96 of them safetensors, 1.56 TB total.
- **Config:**
  - `model_type: kimi_k31` (`KimiK31ForConditionalGeneration`); the text model is `kimi_linear`.
  - `max_position_embeddings` is 1,048,576.
  - The routed experts are `compressed-tensors` `mxfp4-pack-quantized` (group size 32). Everything else is BF16.
  - Needs `--trust-remote-code`.
- **Architecture:**
  - 2.8T total parameters, 104B active per token.
  - 93 layers: 69 KDA (Kimi Delta Attention, which keeps a linear/state cache) and 24 gated DSA layers (MLA plus a DeepSeek Sparse Attention indexer, top-2048).
  - 896 routed experts with 16 active, plus 2 shared experts.
  - Hidden size 7168, 96 attention heads.
  - MoonViT-V2 vision encoder; accepts text and images.
- **It's not Kimi-K3** (`model_type: kimi_k3`). Don't use `/scratch_local/user_data/models/Kimi-K3`, which is jingxinpan's copy of the older model.
- **Serving behaviour:**
  - Sampling is fixed by the model at temperature 1.0 and top_p 0.95; don't hardcode overrides.
  - `reasoning_effort` takes `low`, `high` or `max` (default `max`).
  - In multi-turn use, clients send back the full assistant message, including `reasoning_content` and `tool_calls`.
  - Parsers: `kimi_k3` for both reasoning and tool calls.
- **Speculative draft model:** `RadixArk/Kimi-K3.1-Dflash2-Preview`, revision `ea257b38a8ee`. About 2.78B parameters, BF16, 5.57 GB, Apache-2.0 licensed.

## Memory fit (estimates)

- **16 GPUs is the minimum.** 8 GB200s can't hold it: 1.56 TB / 8 ≈ 195 GB per GPU, more than 185 GiB.
- **Room left at TP16:** about 91 GiB of weights per GPU. At `--mem-fraction-static 0.85` that leaves about 66 GiB for KV cache, KDA state and CUDA graphs.
- **KV cache:**
  - MLA plus indexer KV is about 18.9 KB per token (cookbook figure). TP replicates it and DCP shards it.
  - Plain TP16 holds roughly 2M tokens in total. TP16·DCP16 holds roughly 30M+.
- **KDA state** is about 52 MB per request per rank at TP8 (cookbook figure), and it caps concurrency.
- **DFLASH overhead:** speculative KDA state grows with draft width × running requests, plus about 15% extra KV for the draft.

## Engine choice: SGLang

K3.1 support comes from SGLang's day-0 work, merged to `main` on Nov 4 in PR #48415. PyPI 0.5.23 predates it.

- **SGLang cookbook:** its GB200 "Balanced" shape is **TP16 · DCP16**. It's marked as an untested placeholder that's still in verification.
- **SemiAnalysis InferenceX PR #4804** (merged 2026-11-05) ran K3.1 on GB200 with exactly this shape: 16 GPUs across 4 trays, TP16·DCP16, FP8 KV, DFLASH. That's the closest real-world evidence.
- **vLLM:** its NVIDIA recipe is TP8 on two GB300 trays (288 GB per GPU), which doesn't fit GB200. It has no GB200 or 16-GPU recipe, and its fast decode path requires TP8. Use it only as a fallback, and only in a venv.

## Building the venv (no containers)

1. **Get the source** from github.com/sgl-project/sglang, branch `kimi-k3.1-day-0` (per the cookbook; another source says `kimi-k31-pro`). Check which exists with `git ls-remote`.
2. **Read the pins.** `docker/k31.Dockerfile` in that branch is the authoritative list of pins: sglang-kernel, FlashInfer (0.7.2 in the measured build), sgl-deep-gemm, torch, NCCL, CUDA 13. Read it as text and reproduce the pins with `uv pip`. Don't run Docker.
3. **Mirror `ds.deploy`'s layout:**
   - Put the venv at `kimi.deploy/.venvs/sglang` (gitignored).
   - Install the CUDA 13 toolchain from the `nvidia-*-cu13` wheels and set `CUDA_HOME=<site-packages>/nvidia/cu13`.
   - Put node-local kernel caches (`SGLANG_JIT_CACHE_DIR`, `DG_JIT_CACHE_DIR`, `TRITON_CACHE_DIR` and so on) under `kimi.deploy/.runs`.
4. **Check driver compatibility early.** The driver supports CUDA 13.0, so CUDA 13.x minor-version compatibility should apply. Watch for "driver version is insufficient" errors, or PTX JIT errors.
5. **Replicate the venv.** It must exist at the same absolute path on all 4 nodes. Build it once on c04, then copy it with `tar | ssh` to the same path on the other three, together with the uv-managed Python it points to (`~/.local/share/uv/python/cpython-*`; see `pyvenv.cfg`). c14 has no uv Pythons yet.
6. **Patches:**
   - `glm.deploy` applies an `http_server` middleware patch.
   - `ds.deploy`'s extras (web search, thinking budget, checkpoint overlay) are model-specific.
   - Port any of these only if the user asks.

## Staging the weights (get the user's go-ahead first)

- **Where:** put them on all 4 nodes at `/scratch_local/user_data/yiming/models/Kimi-K3.1-Pro` and `/scratch_local/user_data/yiming/models/Kimi-K3.1-Dflash2-Preview`.
- **How to download:** use `hf download <repo> --revision <rev> --local-dir <dir>`.
- **Use a flat `--local-dir`, not the HF cache.** With `trust_remote_code`, the cache's symlinks make transformers look for `media_utils.py` under `blobs/` and fail (SGLang issue #39568).
- **Download once on c04, then copy to c11, c13 and c14 over `bond0`.** Use parallel `tar | ssh` streams, or rsync with `--rsync-path`.
- **Verify every shard on every node:** compare sha256 against the LFS hashes from the HF API (`/api/models/<repo>/paths-info` or `?blobs=true`).
- **At serve time set `HF_HUB_OFFLINE=1`.** In the InferenceX run, nodes fetching the draft from HF at different speeds broke SGLang's 480 s post-load barrier.

## Stage 1: baseline launch

Run one process per node with `RANK` set to 0–3 as in the node table. Run it in tmux, like `ds41ctl`.

```bash
export SGLANG_K3_SHARDED_MOE=0      # InferenceX fix for a TP16 CUDA-graph crash in the sharded MoE front
export NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1
export GLOO_SOCKET_IFNAME=bond0 NCCL_SOCKET_IFNAME=bond0
export HF_HUB_OFFLINE=1
# plus the CUDA_HOME, PATH, LD_LIBRARY_PATH and cache variables from ds.deploy's launch_environment()

sglang serve --model-path /scratch_local/user_data/yiming/models/Kimi-K3.1-Pro \
  --trust-remote-code \
  --tp-size 16 --dcp-size 16 \
  --nnodes 4 --node-rank $RANK --dist-init-addr 10.78.202.32:20000 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float32 \
  --mem-fraction-static 0.85 \
  --context-length 262144 \
  --max-running-requests 64 \
  --disable-radix-cache \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 \
  --watchdog-timeout 3600 --disable-flashinfer-autotune \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --host 0.0.0.0 --port 30000
```

- **API key:** add one the way `ds.deploy` does, from a new mode-0600 file under `kimi.deploy/.secrets`. Don't reuse `ds.deploy`'s secrets.
- **Context length:** raise it to 1,048,576 once the server is stable.
- **Startup is slow.** Loading 1.5 TB and capturing CUDA graphs can take 30–40 minutes (the K3 report took about 2,280 s to become ready). Size your timeouts for that.
- **Leave these unset:**
  - `--dcp-comm-backend`: GB200 auto-selects `fi_a2a`.
  - `--enable-symm-mem`: it's force-disabled under DCP, and deadlocks with MegaMoE.
  - `--moe-a2a-backend` and `--moe-runner-backend`: the default is the fused MXFP4 TRT-LLM MoE.
  - The DSA backend: the default is FlashMLA.
- **Don't carry over the old Kimi-K3 settings** from jingxinpan's runs: `moonep` all-to-all, EP16, the `deep_gemm` runner, `trtllm_mla`.
- **Check flag names** against the branch's `server_args.py` before the first launch. `--dcp-size`, `--mamba-ssm-dtype` and `--enable-linear-replayssm-spec` were taken from the cookbook and PR text, not tested.

## Stage 2: DFLASH speculative decoding

This roughly halves decode time per token (B300 at concurrency 1: 9.6 ms → 4.89 ms). Add:

```
--speculative-algorithm DFLASH \
--speculative-draft-model-path /scratch_local/user_data/yiming/models/Kimi-K3.1-Dflash2-Preview \
--speculative-draft-attention-backend cutedsl_mla --speculative-draft-kv-cache-dtype fp8_e4m3 \
--speculative-num-draft-tokens 8 \
--speculative-use-block-verification --enable-linear-replayssm-spec
```

- **Width:** use 8 draft tokens for up to about 8 concurrent requests, with `--max-running-requests` of 32 or less. Use 6 for about 24–56 concurrent requests.
- **Expected acceptance length:** 3.0–4.3 at width 5 and 3.4–5.8 at width 8, depending on the dataset.
- **First check the branch has the fix from SGLang PR #48178.** DFLASH combined with DCP stalled at CUDA-graph capture for the K3 recipes.

## Stage 3: prefix cache (HiCache)

For agent workloads with repeated prefixes:
- Drop `--disable-radix-cache`.
- Enable HiCache with `--hicache-io-backend direct --hicache-mem-layout page_first_direct --mamba-max-states-per-path 1`, sized at about 120 GB of host RAM per GPU, as InferenceX did. Each node has about 1.6 TB of RAM free.
- TP16·DCP16, DFLASH and HiCache combined on GB200 has only been run by InferenceX, so add it last.

## Known issues

- **TP16 CUDA-graph crash** in the sharded MoE front: the N=1048 slice GEMM. Fix: `SGLANG_K3_SHARDED_MOE=0`.
- **Kimi-K3 TP16 decode hang on 4×4 GB200** (SGLang issue #39568, sglang 0.5.16):
  - All ranks stall in `process_batch_result_decode` after the first decode step.
  - It didn't reproduce on a newer dev build run with the autotune, watchdog and multithread-load flags above.
  - The fused KDA decode path is suspected.
  - Test for it: 32 concurrent requests of 8K input / 1K output right after startup.
- **DFLASH + DCP graph-capture stall:** see SGLang PR #48178.
- **MegaMoE plus `--enable-symm-mem`** deadlocks.
- **Reading crash logs:** a rank that dies with gloo "connection closed/reset by peer" usually means another rank or the head exited first. Read rank 0's log first. That's how a previous K3 4-node run on this rack failed on Sep 20.

## Public access via frp (approved: remote port 8313)

Added 2026-11-10 after your first stages. The user approved publishing the Kimi API through frp on **remote port 8313**. c02's DeepSeek tunnel uses remote port 8312 and must keep running untouched.

**How c02's tunnel works.** Reference: `reference/ds.deploy/ds41ctl`, lines 41–45 for the constants and 485–560 for `tunnel_state`, `tunnel_status`, `tunnel_start` and `tunnel_stop`.
- `frpc` 0.71.1 connects to the frp server at `ssh.linsyking.com:17000` (129.114.108.73) over TLS, with token auth.
- Its transport goes through a local `gost` HTTP proxy at `127.0.0.1:18080`. gost is started as `gost -L 'http://127.0.0.1:18080?resolver=https://1.1.1.1/dns-query'`, so DNS goes over HTTPS.
- `ds41ctl` handles startup:
  - It runs `frpc verify -c <config>` first.
  - If gost isn't running it starts it with `nohup`, then starts `frpc` the same way.
  - It records state in `.runs/frpc.json` and logs to `.runs/frpc.log`.
  - It refuses to start unless the config's remote-8312 rule targets the configured SGLang host and port, and the API key file exists.
- The rule `deepseek-c02-18000-to-8312` (tcp) forwards server port 8312 to `10.78.202.30:18000`.
  - On the frp server, the API is reachable only locally, at `http://127.0.0.1:8312/v1` on `ssh.linsyking.com`.
  - SGLang's API key protects it.

**Kimi's tunnel on c04:**
- **Config:** `kimi.deploy/.secrets/frpc.toml`, mode 600 and gitignored, already in place.
  - It uses the same server, TLS and token as c02's, with one rule: `kimi-c04-30000-to-8313` (tcp), forwarding `10.78.202.32:30000` to remote port 8313.
  - `frpc verify` passes.
  - Never copy the token anywhere else: not into logs, docs or commits.
- **Binaries:** `/scratch_local/user_data/yiming/.local/bin/frpc` and `.../gost` on c04 are byte-identical to c02's.
- **Run a separate gost and frpc on c04.** `127.0.0.1:18080` is free there.
  - Don't edit, restart or signal c02's `frpc` or `gost`.
  - Never reuse the proxy name `deepseek-c02-18000-to-8312`; the frp server keys rules by name.
- c04 also resolves `ssh.linsyking.com` directly. Keep the gost transport anyway, to match the setup that's known to work.
- **Add `tunnel start|stop|status` to `kimictl`, modelled on `ds41ctl`:**
  - Verify the config, and refuse to start unless the 8313 rule targets `serving.json`'s `bind_host:port` and `.secrets/api-key` exists with API-key auth enabled on the server.
  - Start gost if it isn't running, then start `frpc -c .secrets/frpc.toml` detached, so it survives SSH disconnects.
  - Record state in `.runs/frpc.json` and log to `.runs/frpc.log`.
  - Stop it through the recorded process group.
  - No automatic restart.
  - Like DeepSeek's tunnel, it needs restarting by hand after a reboot.
- **When to start it:** only while the server is up, validated and requiring the API key.
- **Then:** confirm `start proxy success` for `kimi-c04-30000-to-8313` in `.runs/frpc.log`, and tell the user:
  - the server-local URL, `http://127.0.0.1:8313/v1` on `ssh.linsyking.com`
  - the served model name, `kimi-k3.1-pro`
  - You can't test from the frp server side yourself, so report the log lines.

## Validation checklist

1. **Pre-flight:**
   - `check_rack.py` on the 4 nodes shows HEALTHY, 72/72 links, recovery action None.
   - GPUs show 0 MiB used, and no other users are present.
2. **Environment:**
   - `import sglang` works in the venv, and the K3.1 model class is registered.
   - One launch with `NCCL_DEBUG=INFO` shows MNNVL/NVLS in use across trays, not socket transport.
3. **Function:**
   - `/health` and `/v1/models` respond.
   - Chat works with `reasoning_effort` set to low and to max.
   - A tool call works.
   - An image input works.
   - A ~128K-token needle-in-a-haystack test passes.
   - The 32-concurrency stress test runs without hanging.
4. **Performance:**
   - Run `sglang bench_serving` at concurrency 1, 8, 32 and 64, and record TTFT, TPOT and throughput.
   - In stage 2, also record the DFLASH acceptance length.
   - The only published reference numbers are for B300, not GB200: 1P1D decode at 8K context and batch 64 gives 2,583 tok/s with TPOT p90 of 26 ms.

## Deliverables (mirror `ds.deploy`)

- **In `kimi.deploy/`:**
  - A controller script that handles start, stop, status and logs across the 4 nodes, using tmux sessions.
    - It refuses to start if GPUs are busy or a recovery action isn't None.
    - It never resets GPUs.
  - `serving.json`, `README.md`, `DEPLOYMENT_STATUS.md`.
  - A `.gitignore` covering `.venvs/`, `.runs/`, `logs/`, `.secrets/`, `models/` and `*token*`.
- **Git:** commit to the local `kimi.deploy` repo. No remote is configured, so ask before adding one or pushing.

## Sources

- Model card: https://huggingface.co/moonshotai/Kimi-K3.1-Pro
- SGLang cookbook: https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro
- InferenceX GB200 run: https://github.com/SemiAnalysisAI/InferenceX/pull/4804
- K3 TP16 GB200 hang: https://github.com/sgl-project/sglang/issues/39568
- DFLASH + DCP fix: https://github.com/sgl-project/sglang/pull/48178
- Draft model: https://huggingface.co/RadixArk/Kimi-K3.1-Dflash2-Preview
- vLLM recipe (fallback reference): https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro
- vLLM day-0 blog: https://vllm.ai/blog/2026-11-04-kimi-k31-pro
