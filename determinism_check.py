#!/usr/bin/env python3
"""Does the same prefill give the same logprobs twice? Scores one fixed token sequence several times,
each on a flushed cache, and counts positions whose logprob moves or whose top-1 changes. A correct
engine moves nearly nothing (batch-shape numerics at most); a race or a stale buffer moves a few
positions by whole nats. The sequence (an OBJ greedy output with its prompt) is saved on first use,
so later runs and other configurations score the same tokens. Stdlib only; on c04.

    determinism_check.py [--repeats 4] [--sequence .runs/determinism/sequence.json]"""
import argparse
import datetime
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PROMPT = ('Output a Wavefront OBJ file for a UV sphere of radius 1: 24 segments, 16 rings, "v x y z" lines with '
          '4 decimals (north pole, 15 rings of 24, south pole), then all faces with 1-based indices. '
          'Output only the file contents, no code fence, no commentary.')


def post(base, path, key, body=None):
    request = urllib.request.Request(base + path, data=b'' if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with OPENER.open(request, timeout=1800) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--sequence', type=Path, default=ROOT / '.runs/determinism/sequence.json')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'serving.json').read_text())
    key = (ROOT / '.secrets/api-key').read_text().strip()
    admin = (ROOT / '.secrets/admin-api-key').read_text().strip()
    base = f'http://{cfg["bind_host"]}:{cfg["port"]}'
    if not args.sequence.exists():
        out = json.loads(post(base, '/v1/chat/completions', key, {
            'model': cfg['served_model_name'], 'messages': [{'role': 'user', 'content': PROMPT}],
            'temperature': 0, 'max_tokens': 20000, 'reasoning_effort': 'low', 'return_token_ids': True}))
        choice = out['choices'][0]
        ids = (choice.get('prompt_token_ids') or out['prompt_token_ids']) + choice['response_token_ids']
        args.sequence.parent.mkdir(parents=True, exist_ok=True)
        args.sequence.write_text(json.dumps({'ids': ids}))
    ids = json.loads(args.sequence.read_text())['ids']
    start = 64
    runs = []
    for _ in range(args.repeats):
        post(base, '/flush_cache?timeout=120', admin)
        meta = json.loads(post(base, '/generate', key, {
            'input_ids': ids, 'sampling_params': {'max_new_tokens': 1, 'temperature': 0}, 'return_logprob': True,
            'logprob_start_len': start, 'top_logprobs_num': 1}))['meta_info']
        runs.append([(entry[0], top[0][1]) for entry, top in
                     zip(meta['input_token_logprobs'][1:], meta['input_top_logprobs'][1:])])
    first, report = runs[0], []
    for number, run in enumerate(runs[1:], 1):
        moved = [(abs(a[0] - b[0]), index) for index, (a, b) in enumerate(zip(first, run))]
        report.append({'run': number, 'positions': len(moved), 'identical': sum(m == 0 for m, _ in moved),
                       'top1_changed': sum(a[1] != b[1] for a, b in zip(first, run)),
                       'over_0.1': sum(m > 0.1 for m, _ in moved), 'over_1': sum(m > 1 for m, _ in moved),
                       'max': round(max(moved)[0], 3), 'worst_positions': [i + start + 1 for _, i in sorted(moved)[-5:]]})
        print(json.dumps(report[-1]), flush=True)
    try:
        build = json.loads((ROOT / '.venvs' / cfg.get('venv', 'sglang') / 'kimi-build.json').read_text())
    except (OSError, ValueError):
        build = {}
    out = ROOT / '.runs/determinism' / f'{datetime.datetime.now():%Y%m%d-%H%M%S}.json'
    out.write_text(json.dumps({'sglang_commit': build.get('sglang_commit'), 'speculative': cfg.get('speculative'),
                               'extra_env': cfg.get('extra_env'), 'extra_args': cfg.get('extra_args'),
                               'tokens': len(ids), 'report': report}, indent=1) + '\n')
    print(f'{len(ids)} tokens; results in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
