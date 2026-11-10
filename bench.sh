#!/usr/bin/env bash
# bench_serving against the running endpoint, run on c04, at each concurrency in turn.
#   DATASET=random (default): fixed 8K input / 1K output built from ShareGPT text through native
#     /generate. The random dataset repeats a short prompt up to the input length, so this text is
#     repetitive and inflates DFLASH acceptance; use it for comparable speed numbers.
#   DATASET=sharegpt: real ShareGPT first turns as chat requests (reasoning at the model's default)
#     that stop on their own, output capped at OUTPUT_LEN; use it for acceptance on natural text.
# Acceptance length is the run's generated tokens per verify step, from rank 0's /metrics counters.
# Results: .runs/bench/<label>-c<N>.{jsonl,log,accept}
#
#   bench.sh <label> [concurrency ...]      default concurrencies: 1 8 32 64
set -euo pipefail

ROOT=/scratch_local/user_data/yiming/kimi.deploy
SCRATCH=/scratch_local/user_data/yiming
SHAREGPT=$ROOT/.runs/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
OUT=$ROOT/.runs/bench
LABEL=${1:?usage: bench.sh <label> [concurrency ...]}
shift
if [ $# -gt 0 ]; then CONCURRENCY=("$@"); else CONCURRENCY=(1 8 32 64); fi
DATASET=${DATASET:-random}
INPUT_LEN=${INPUT_LEN:-8192}
OUTPUT_LEN=${OUTPUT_LEN:-1024}

read -r HOST PORT MODEL MODEL_PATH < <(python3 -c '
import json, sys; c = json.load(open(sys.argv[1]))
print(c["bind_host"], c["port"], c["served_model_name"], c["model_path"])' "$ROOT/serving.json")

export HF_HOME=$SCRATCH/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1
if [ ! -s "$SHAREGPT" ]; then
    "$SCRATCH/.local/bin/hf" download anon8231489123/ShareGPT_Vicuna_unfiltered \
        ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir "$(dirname "$SHAREGPT")"
fi
mkdir -p "$OUT"
OPENAI_API_KEY=$(cat "$ROOT/.secrets/api-key")
export OPENAI_API_KEY HF_HUB_OFFLINE=1

case $DATASET in
    random) WORKLOAD=(--backend sglang --dataset-name random --random-input-len "$INPUT_LEN"
                      --random-output-len "$OUTPUT_LEN" --random-range-ratio 1.0) ;;
    sharegpt) WORKLOAD=(--backend sglang-oai-chat --dataset-name sharegpt --sharegpt-output-len "$OUTPUT_LEN"
                        --disable-ignore-eos) ;;
    *) echo "DATASET must be random or sharegpt" >&2; exit 2 ;;
esac

counter() {  # sum of a counter over its label sets on rank 0's /metrics
    curl -s --noproxy '*' -m 30 -H "Authorization: Bearer $OPENAI_API_KEY" "http://$HOST:$PORT/metrics" |
        awk -v name="$1" 'index($1, name "{") == 1 || $1 == name {sum += $2} END {printf "%.0f\n", sum}'
}

for c in "${CONCURRENCY[@]}"; do
    prompts=$(( c * 4 > 8 ? c * 4 : 8 ))
    name=$OUT/$LABEL-c$c
    echo "== $LABEL: $DATASET, concurrency $c, $prompts prompts"
    tokens=$(counter sglang:generation_tokens_total); verifies=$(counter sglang:spec_verify_calls_total)
    # A run that outlives this bound is treated as a hang; the server itself is left alone.
    timeout 3600 "$ROOT/.venvs/sglang/bin/python" -m sglang.benchmark.serving \
        --base-url "http://$HOST:$PORT" --model "$MODEL" --tokenizer "$MODEL_PATH" \
        --dataset-path "$SHAREGPT" "${WORKLOAD[@]}" --num-prompts "$prompts" --max-concurrency "$c" \
        --output-file "$name.jsonl" > "$name.log" 2>&1 || {
        echo "bench at concurrency $c failed or timed out; see $name.log" >&2
        exit 1
    }
    echo "$(( $(counter sglang:generation_tokens_total) - tokens )) $(( $(counter sglang:spec_verify_calls_total) - verifies ))" > "$name.accept"
    sed -n '/Serving Benchmark Result/,/^=\+$/p' "$name.log" | tail -n +2
done

python3 - "$OUT" "$LABEL" "${CONCURRENCY[@]}" <<'EOF'
import json, sys
out, label, *levels = sys.argv[1:]
print('| Concurrency | Requests | TTFT mean / p90 (ms) | TPOT mean / p90 (ms) | ITL p90 (ms) | Output tok/s | Total tok/s | Accept length |')
print('|---:|---:|---:|---:|---:|---:|---:|---:|')
for c in levels:
    r = json.loads(open(f'{out}/{label}-c{c}.jsonl').read().splitlines()[-1])
    tokens, verifies = map(int, open(f'{out}/{label}-c{c}.accept').read().split())
    accept = f'{tokens / verifies:.2f}' if verifies else '-'
    print(f'| {c} | {r["completed"]} | {r["mean_ttft_ms"]:,.0f} / {r["p90_ttft_ms"]:,.0f} | '
          f'{r["mean_tpot_ms"]:.1f} / {r["p90_tpot_ms"]:.1f} | {r["p90_itl_ms"]:.1f} | '
          f'{r["output_throughput"]:,.0f} | {r["total_throughput"]:,.0f} | {accept} |')
EOF
