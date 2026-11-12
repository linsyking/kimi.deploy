#!/usr/bin/env python3
"""Does the decode path emit tokens a prefill of the same text would not? Greedy, so it is exact.

Each request greedily decodes an OBJ sphere (the DFLASH path, several requests in flight so request
rows are reused under load). After one cache flush, each output is scored in a single prefill pass:
an emitted token that is not the prefill's top-1 is a disagreement, and one whose logprob trails
the top-1 by a nat or more is a large one. Numerics differ between the two paths, so near-ties
flip; large ones should not. Also counts face lines with a negative index, the visible symptom.
Stdlib only; on c04. Reads the keys in-process.

    decode_drift_check.py [--requests 16] [--concurrency 8]"""
import argparse
import concurrent.futures
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
LARGE = 1.0


def call(base, path, key, body=None, timeout=1800):
    request = urllib.request.Request(base + path, data=b'' if body is None else json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with OPENER.open(request, timeout=timeout) as response:
        return response.status, response.read()


def generate(base, cfg, key):
    _, raw = call(base, '/v1/chat/completions', key, {
        'model': cfg['served_model_name'], 'messages': [{'role': 'user', 'content': PROMPT}],
        'temperature': 0, 'max_tokens': 20000, 'reasoning_effort': 'low', 'return_token_ids': True})
    out = json.loads(raw)
    choice = out['choices'][0]
    return {'prompt_ids': choice.get('prompt_token_ids') or out.get('prompt_token_ids'),
            'gen_ids': choice.get('response_token_ids'), 'text': choice['message'].get('content') or ''}


def score(base, key, run):
    """Prefill the prompt and the emitted tokens; compare each emitted token with the prefill's top-1."""
    prompt, gen = run['prompt_ids'], run['gen_ids']
    _, raw = call(base, '/generate', key, {
        'input_ids': prompt + gen, 'sampling_params': {'max_new_tokens': 1, 'temperature': 0},
        'return_logprob': True, 'logprob_start_len': len(prompt) - 1, 'top_logprobs_num': 1,
        'return_text_in_logprobs': True})
    meta = json.loads(raw)['meta_info']
    entries, tops = meta['input_token_logprobs'][1:], meta['input_top_logprobs'][1:]
    disagreements = []
    for index, (entry, top) in enumerate(zip(entries, tops)):
        if top and entry[1] != top[0][1] and top[0][0] - entry[0] > 1e-3:
            context = ''.join(e[2] or '' for e in entries[max(0, index - 12):index])
            disagreements.append({'index': index, 'emitted': entry[2], 'top1': top[0][2],
                                  'margin': round(top[0][0] - entry[0], 3), 'context': context[-60:]})
    return len(entries), disagreements


def face_minus(text):
    return [line for line in text.splitlines() if line.startswith('f ') and '-' in line]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--requests', type=int, default=16)
    parser.add_argument('--concurrency', type=int, default=8)
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'serving.json').read_text())
    key = (ROOT / '.secrets/api-key').read_text().strip()
    admin = (ROOT / '.secrets/admin-api-key').read_text().strip()
    base = f'http://{cfg["bind_host"]}:{cfg["port"]}'
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        runs = list(pool.map(lambda _: generate(base, cfg, key), range(args.requests)))
    # Score on a cold cache: a radix hit would reuse the decode path's own KV for the emitted tokens.
    status, _ = call(base, '/flush_cache?timeout=120', admin, timeout=180)
    print(f'flush_cache: HTTP {status}', flush=True)
    totals = {'tokens': 0, 'disagreements': 0, 'large': 0, 'face_minus_lines': 0}
    for number, run in enumerate(runs):
        if not (run['prompt_ids'] and run['gen_ids']):
            print(f'#{number}: no token ids returned')
            continue
        count, disagreements = score(base, key, run)
        large = [d for d in disagreements if d['margin'] >= LARGE]
        minus = face_minus(run['text'])
        run.update(scored=count, disagreements=disagreements, large=len(large), face_minus=minus)
        totals['tokens'] += count
        totals['disagreements'] += len(disagreements)
        totals['large'] += len(large)
        totals['face_minus_lines'] += len(minus)
        print(f'#{number}: {count} tokens, {len(disagreements)} not prefill top-1, {len(large)} by >= {LARGE} nat; '
              f'{len(minus)} face lines with "-"', flush=True)
        for item in sorted(large, key=lambda d: -d['margin'])[:3]:
            print(f'   {item["margin"]:5.2f} emitted {item["emitted"]!r} top-1 {item["top1"]!r} after {item["context"]!r}')
    try:
        build = json.loads((ROOT / '.venvs' / cfg.get('venv', 'sglang') / 'kimi-build.json').read_text())
    except (OSError, ValueError):
        build = {}
    out = ROOT / '.runs/decode-drift' / f'{datetime.datetime.now():%Y%m%d-%H%M%S}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'sglang_commit': build.get('sglang_commit'), 'speculative': cfg.get('speculative'),
                               'weights': 'abliterated' if cfg.get('checkpoint_overlay') else 'official',
                               'concurrency': args.concurrency, 'totals': totals,
                               'runs': [{k: v for k, v in run.items() if k not in ('prompt_ids', 'gen_ids')}
                                        for run in runs]}, indent=1, ensure_ascii=False) + '\n')
    print(f'totals {totals}; results in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
