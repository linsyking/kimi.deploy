#!/usr/bin/env bash
# Build .venvs/sglang on c04 from the pinned kimi-k3.1-day-0 checkout. Reproduces the Python
# layer of docker/k31.Dockerfile (its nightly base image's dependency solve plus the K3.1 steps)
# with uv and pip CUDA 13.0 wheels; no containers. Copy the result with replicate_venv.sh.
#
#   ./build_venv.sh            install from requirements-sglang.lock
#   ./build_venv.sh --relock   re-resolve the lock from the checkout's pyproject first
set -euo pipefail

ROOT=/scratch_local/user_data/yiming/kimi.deploy
LOCAL=/scratch_local/user_data/yiming/.local
UV=$LOCAL/bin/uv
RUST_BIN=$LOCAL/opt/rust/bin
SRC=$ROOT/.src/sglang
VENV=$ROOT/.venvs/sglang
PY=$VENV/bin/python
BUILD=$ROOT/.runs/build
LOCK=$ROOT/requirements-sglang.lock

SGLANG_REPO=https://github.com/sgl-project/sglang.git
SGLANG_BRANCH=kimi-k3.1-day-0
SGLANG_COMMIT=8b180ca9e07e108d0600096aa1c3adda9e72513d
PYTHON_VERSION=3.12.13
FLASHINFER_VERSION=0.7.3
NCCL_VERSION=2.30.7
TORCH_INDEX=https://download.pytorch.org/whl/cu130

# Non-interactive SSH sets HOME to the NFS home; keep uv's cache and Pythons on scratch.
export UV_CACHE_DIR=/scratch_local/user_data/yiming/.cache/uv
export UV_PYTHON_INSTALL_DIR=$LOCAL/share/uv/python
export UV_PYTHON_PREFERENCE=only-managed
# The K3.1 FlashInfer steps rewrite installed files in place; hardlinks would carry those
# edits into uv's cache. Copies also keep the venv self-contained for replication.
export UV_LINK_MODE=copy
export TMPDIR=$BUILD/tmp
mkdir -p "$TMPDIR"

step() { printf '\n==> %s\n' "$*"; }
pip_install() { "$UV" pip install --python "$PY" "$@"; }

step "Source: $SGLANG_BRANCH @ ${SGLANG_COMMIT:0:12}"
if [ ! -d "$SRC/.git" ]; then
    git clone --filter=blob:none -b "$SGLANG_BRANCH" "$SGLANG_REPO" "$SRC"
    git -C "$SRC" checkout -q --detach "$SGLANG_COMMIT"
fi
head=$(git -C "$SRC" rev-parse HEAD)
if [ "$head" != "$SGLANG_COMMIT" ]; then
    echo "error: $SRC is at $head, expected $SGLANG_COMMIT; check it out or update SGLANG_COMMIT" >&2
    exit 1
fi
if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]; then
    echo "error: $SRC has local modifications" >&2
    exit 1
fi

step "Venv: CPython $PYTHON_VERSION at $VENV"
[ -x "$PY" ] || "$UV" venv --seed --python "$PYTHON_VERSION" "$VENV"

if [ "${1:-}" = "--relock" ] || [ ! -f "$LOCK" ]; then
    step "Resolve $LOCK"
    "$UV" pip compile "$SRC/python/pyproject.toml" -c "$ROOT/constraints-cuda130.txt" \
        --python "$PY" --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
        --no-annotate --custom-compile-command "./build_venv.sh --relock" -o "$LOCK"
fi

step "Dependencies from the lock (the image's pyproject solve, cu130 torch index)"
pip_install -r "$LOCK" -c "$ROOT/constraints-cuda130.txt" \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match

step "Image extras: NCCL $NCCL_VERSION for DeepEP, FlashInfer cubin + cu130 jit-cache $FLASHINFER_VERSION"
pip_install --reinstall --no-deps "nvidia-nccl-cu13==$NCCL_VERSION"
pip_install --no-deps "flashinfer-cubin==$FLASHINFER_VERSION" --index-url https://flashinfer.ai/whl
pip_install --no-deps "flashinfer-jit-cache==$FLASHINFER_VERSION+cu130" --index-url https://flashinfer.ai/whl/cu130

step "SGLang from source with its Rust extensions (non-editable, so the venv is self-contained)"
pip_install "setuptools>=61" "setuptools-rust>=1.11" "setuptools-scm>=8" wheel
PATH=$RUST_BIN:$PATH CARGO_HOME=$BUILD/cargo CARGO_BUILD_JOBS=$(nproc) \
    pip_install --no-deps --no-build-isolation --reinstall-package sglang "$SRC/python"

SITE=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
CUDA=$SITE/nvidia/cu13
[ -x "$CUDA/bin/nvcc" ] || { echo "error: no nvcc at $CUDA/bin" >&2; exit 1; }

step "K3.1 FlashInfer DynBlock: 1024-expert routing header, rebuilt jit-cache modules"
CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDA_HOME=$CUDA CUDA_PATH=$CUDA \
PATH=$CUDA/bin:$VENV/bin:$PATH LD_LIBRARY_PATH=$CUDA/lib:${LD_LIBRARY_PATH:-} \
    bash -c "set -e; '$PY' -I '$SRC/docker/k31-image/patch_flashinfer.py'; \
             '$PY' -I '$SRC/docker/k31-image/build_flashinfer_dynblock.py'"

cat > "$VENV/kimi-build.json" <<EOF
{"sglang_repo": "$SGLANG_REPO", "sglang_branch": "$SGLANG_BRANCH", "sglang_commit": "$SGLANG_COMMIT",
 "python": "$PYTHON_VERSION", "flashinfer": "$FLASHINFER_VERSION", "nccl": "$NCCL_VERSION",
 "built_on": "$(hostname -s)", "built_at": "$(date -u +%FT%TZ)"}
EOF

step "Verify"
"$PY" -I "$ROOT/verify_venv.py"
