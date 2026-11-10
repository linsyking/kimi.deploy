#!/usr/bin/env bash
# bench_serving against the running endpoint, run on c04: fixed 8K input / 1K output built from
# ShareGPT text (the random dataset), at each concurrency in turn. Native /generate is used so
# DFLASH acceptance lengths are reported. Results: .runs/bench/<label>-c<N>.{jsonl,log}
#
#   bench.sh <label> [concurrency ...]      default concurrencies: 1 8 32 64
set -euo pipefail

ROOT=/scratch_local/user_data/yiming/kimi.deploy
SCRATCH=/scratch_local/user_data/yiming
DATASET=$ROOT/.runs/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
OUT=$ROOT/.runs/bench
LABEL=${1:?usage: bench.sh <label> [concurrency ...]}
shift
if [ $# -gt 0 ]; then CONCURRENCY=("$@"); else CONCURRENCY=(1 8 32 64); fi
INPUT_LEN=${INPUT_LEN:-8192}
OUTPUT_LEN=${OUTPUT_LEN:-1024}

read -r HOST PORT MODEL MODEL_PATH < <(python3 -c '
import json, sys; c = json.load(open(sys.argv[1]))
print(c["bind_host"], c["port"], c["served_model_name"], c["model_path"])' "$ROOT/serving.json")

export HF_HOME=$SCRATCH/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1
if [ ! -s "$DATASET" ]; then
    "$SCRATCH/.local/bin/hf" download anon8231489123/ShareGPT_Vicuna_unfiltered \
        ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir "$(dirname "$DATASET")"
fi
mkdir -p "$OUT"
OPENAI_API_KEY=$(cat "$ROOT/.secrets/api-key")
export OPENAI_API_KEY HF_HUB_OFFLINE=1

for c in "${CONCURRENCY[@]}"; do
    prompts=$(( c * 4 > 8 ? c * 4 : 8 ))
    name=$OUT/$LABEL-c$c
    echo "== $LABEL: concurrency $c, $prompts prompts, ${INPUT_LEN} in / ${OUTPUT_LEN} out"
    # A run that outlives this bound is treated as a hang; the server itself is left alone.
    timeout 3600 "$ROOT/.venvs/sglang/bin/python" -m sglang.benchmark.serving \
        --backend sglang --base-url "http://$HOST:$PORT" --model "$MODEL" --tokenizer "$MODEL_PATH" \
        --dataset-name random --dataset-path "$DATASET" --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" --random-range-ratio 1.0 --num-prompts "$prompts" \
        --max-concurrency "$c" --output-file "$name.jsonl" > "$name.log" 2>&1 || {
        echo "bench at concurrency $c failed or timed out; see $name.log" >&2
        exit 1
    }
    sed -n '/Serving Benchmark Result/,/^=\+$/p' "$name.log" | tail -n +2
done
