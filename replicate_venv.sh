#!/usr/bin/env bash
# Copy .venvs/sglang from c04 to the worker nodes at the same absolute path with rsync, plus the
# uv-managed CPython its pyvenv.cfg points to where a node lacks it, then verify on each node.
#
#   ./replicate_venv.sh                  c11, c13 and c14 in parallel
#   ./replicate_venv.sh 10.78.202.42     one node
#
# KIMI_VENV copies another build_venv.sh build. While the service runs, rsync on both ends is
# bound to Grace memory (its page cache would otherwise land in HBM) and the GPU check is skipped
# unless KIMI_VERIFY_GPU=1.
set -euo pipefail

ROOT=/scratch_local/user_data/yiming/kimi.deploy
RSYNC=/scratch_local/user_data/yiming/.local/bin/rsync
VENV=${KIMI_VENV:-$ROOT/.venvs/sglang}
NUMACTL=/scratch_local/user_data/yiming/.local/opt/sysutils/bin/numactl
# Grace's NUMA nodes are the ones with CPUs; the GPUs' HBM shows up as CPU-less nodes.
GRACE_NODES='for n in /sys/devices/system/node/node[0-9]*; do [ -n "$(cat $n/cpulist)" ] && echo ${n##*node}; done | paste -sd,'
VERIFY_GPU=${KIMI_VERIFY_GPU:-0}
PYDIR=$(dirname "$(sed -n 's/^home = //p' "$VENV/pyvenv.cfg")")
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
WORKERS=("$@")
[ ${#WORKERS[@]} -gt 0 ] || WORKERS=(10.78.202.39 10.78.202.41 10.78.202.42)

[ -f "$VENV/kimi-build.json" ] || { echo "error: $VENV is not a finished build_venv.sh build" >&2; exit 1; }
local_python=$(sha256sum "$PYDIR/bin/python3.12" | cut -d' ' -f1)

replicate() {
    local node=$1
    $SSH "$node" "test -x $RSYNC" || { echo "no rsync at $RSYNC; provision the node first" >&2; return 1; }
    remote_python=$($SSH "$node" "sha256sum $PYDIR/bin/python3.12 2>/dev/null | cut -d' ' -f1")
    if [ -z "$remote_python" ]; then
        echo "copying $PYDIR"
        $RSYNC -a --mkpath -e "$SSH" --rsync-path="$RSYNC" "$PYDIR/" "$node:$PYDIR/"
    elif [ "$remote_python" != "$local_python" ]; then
        echo "error: $PYDIR differs from c04's; refusing to overwrite it" >&2
        return 1
    fi
    local here there
    here=$(bash -c "$GRACE_NODES")
    there=$($SSH "$node" "$GRACE_NODES")
    [ -n "$here" ] && [ -n "$there" ] || { echo "no Grace NUMA nodes found" >&2; return 1; }
    "$NUMACTL" --membind="$here" $RSYNC -a --delete --mkpath -e "$SSH" \
        --rsync-path="$NUMACTL --membind=$there $RSYNC" "$VENV/" "$node:$VENV/"
    $RSYNC -a -e "$SSH" --rsync-path="$RSYNC" "$ROOT/verify_venv.py" "$node:$ROOT/"
    $SSH "$node" "KIMI_VENV=$VENV $VENV/bin/python -I $ROOT/verify_venv.py $([ "$VERIFY_GPU" = 1 ] && echo --gpu)"
}

pids=()
for node in "${WORKERS[@]}"; do
    replicate "$node" > >(sed -u "s/^/[$node] /") 2>&1 &
    pids+=($!)
done
failed=0
for i in "${!pids[@]}"; do
    wait "${pids[$i]}" || { echo "[${WORKERS[$i]}] FAILED" >&2; failed=1; }
done
exit $failed
