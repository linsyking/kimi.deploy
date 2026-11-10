#!/usr/bin/env bash
# Stage a pinned Hugging Face snapshot on all four nodes: download it once on c04 into a flat
# --local-dir (not the HF cache, whose symlinks break trust_remote_code; sglang#39568), rsync it
# to c11, c13 and c14 over bond0, and verify every file on every node against the HF hashes.
#
#   ./stage_weights.sh download   pro|draft                 on c04, then verify there
#   ./stage_weights.sh distribute pro|draft [IPs...]        c04 -> workers (default c11 c13 c14)
#   ./stage_weights.sh verify     pro|draft [IPs...]        default: all four nodes
set -euo pipefail

ROOT=/scratch_local/user_data/yiming/kimi.deploy
LOCAL=/scratch_local/user_data/yiming/.local
RSYNC=$LOCAL/bin/rsync
HF=$LOCAL/bin/hf
MODELS=/scratch_local/user_data/yiming/models
RUNS=$ROOT/.runs/weights
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10"
# AES-GCM uses the Grace cores' crypto extensions; one stream is CPU-bound well below bond0.
RSYNC_SSH="$SSH -c aes128-gcm@openssh.com"
STREAMS=${STREAMS:-8}
ALL=(10.78.202.32 10.78.202.39 10.78.202.41 10.78.202.42)
WORKERS=(10.78.202.39 10.78.202.41 10.78.202.42)

KIND=${2:-}
case "$KIND" in
    pro)   REPO=moonshotai/Kimi-K3.1-Pro;           NAME=Kimi-K3.1-Pro ;;
    draft) REPO=RadixArk/Kimi-K3.1-Dflash2-Preview; NAME=Kimi-K3.1-Dflash2-Preview ;;
    *) echo "usage: $0 download|distribute|verify pro|draft [node IPs...]" >&2; exit 2 ;;
esac
MANIFEST=$ROOT/manifests/$NAME.manifest.json
REVISION=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["revision"])' "$MANIFEST")
DIR=$MODELS/$NAME
NODES=("${@:3}")
mkdir -p "$RUNS"

require_c04() {
    [ "$(hostname -s)" = s04-p1-dgx-02-c04 ] || { echo "error: run this on c04" >&2; exit 1; }
}

download() {
    require_c04
    mkdir -p "$DIR"
    HF_HOME=/scratch_local/user_data/yiming/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1 \
    HF_XET_HIGH_PERFORMANCE=1 "$HF" download "$REPO" --revision "$REVISION" --local-dir "$DIR" \
        --max-workers 16
    python3 -I "$ROOT/verify_weights.py" "$DIR"
}

distribute() {
    require_c04
    [ -f "$DIR/.kimi-verified.json" ] || { echo "error: verify $DIR on c04 first" >&2; exit 1; }
    [ ${#NODES[@]} -gt 0 ] || NODES=("${WORKERS[@]}")
    # Deal the files, largest first, round-robin into $STREAMS lists of similar total size.
    local lists=$RUNS/$NAME.streams
    rm -rf "$lists" && mkdir -p "$lists"
    (cd "$DIR" && find . -type f ! -path './.cache/*' ! -name .kimi-verified.json -printf '%s %P\n') |
        sort -rn | awk -v n="$STREAMS" -v d="$lists" '{ sub(/^[0-9]+ /, ""); print > (d "/" (NR - 1) % n) }'
    local pids=() labels=()
    for node in "${NODES[@]}"; do
        $SSH "$node" "mkdir -p $DIR"
        for list in "$lists"/*; do
            "$RSYNC" -a --partial-dir=.rsync-partial -e "$RSYNC_SSH" --rsync-path="$RSYNC" \
                --files-from="$list" "$DIR/" "$node:$DIR/" > "$RUNS/rsync-$NAME-$node-$(basename "$list").log" 2>&1 &
            pids+=($!) labels+=("$node stream $(basename "$list")")
        done
    done
    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "FAILED: ${labels[$i]} (see $RUNS)" >&2; failed=1; }
    done
    [ $failed -eq 0 ] && echo "Copied $NAME to ${NODES[*]}; run: $0 verify $KIND ${NODES[*]}"
    return $failed
}

verify() {
    [ ${#NODES[@]} -gt 0 ] || NODES=("${ALL[@]}")
    local pids=()
    for node in "${NODES[@]}"; do
        (
            $SSH "$node" "mkdir -p $ROOT/manifests"
            "$RSYNC" -a -e "$SSH" --rsync-path="$RSYNC" "$ROOT/verify_weights.py" "$node:$ROOT/"
            "$RSYNC" -a -e "$SSH" --rsync-path="$RSYNC" "$MANIFEST" "$node:$ROOT/manifests/"
            $SSH "$node" "python3 -I $ROOT/verify_weights.py $DIR"
        ) 2>&1 | sed -u "s/^/[$node] /" &
        pids+=($!)
    done
    local failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    return $failed
}

case "$1" in
    download) download ;;
    distribute) distribute ;;
    verify) verify ;;
    *) echo "usage: $0 download|distribute|verify pro|draft [node IPs...]" >&2; exit 2 ;;
esac
