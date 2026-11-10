#!/usr/bin/env python3
"""Multi-turn prefix-reuse benchmark against the running endpoint, run on c04 (stdlib only).

Each session opens with its own long document and asks a question per turn; turn k resends the
whole conversation, so every turn after the first repeats a long prefix. Turns run turn-major
(turn 1 of every session, then turn 2, ...) so a session is revisited only after all the others,
the reuse distance that decides whether a prefix cache, and which tier of it, still holds it.
Reports TTFT per turn, plus prefix-cache counters from /metrics when the server exports them.
Requests carry no sampling overrides. Results go to .runs/bench/<label>.json.

    prefix_bench.py <label> [--sessions 96] [--turns 3] [--doc-tokens 24000] [--concurrency 8]"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import random
import re
import statistics
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
QUESTIONS = ['Which town appears most often in the records above? Answer in one short sentence.',
             'Name one year in which a ferry crossing was recorded. Answer in one short sentence.',
             'How many visitors did the largest season count? Answer in one short sentence.',
             'Which subject was inspected first? Answer in one short sentence.']
WORDS = ['harbour', 'tram', 'market', 'station', 'archive', 'ferry', 'glassworks', 'orchard', 'lock', 'press']


class Client:
    def __init__(self):
        cfg = json.loads((ROOT / 'serving.json').read_text())
        self.base = f'http://{cfg["bind_host"]}:{cfg["port"]}'
        self.model = cfg['served_model_name']
        self.headers = {'Authorization': 'Bearer ' + (ROOT / '.secrets/api-key').read_text().strip(),
                        'Content-Type': 'application/json'}

    def metrics(self):
        """Counter totals, split by cache source or host pool. Series that carry a tp_rank label
        repeat one logical count per rank (HiCache backup, load-back), so only TP0's is kept."""
        req = urllib.request.Request(self.base + '/metrics', headers=self.headers)
        totals = {}
        with OPENER.open(req, timeout=30) as response:
            for line in response.read().decode().splitlines():
                if line.startswith('sglang:') and not line.startswith('#'):
                    name, _, value = line.rpartition(' ')
                    key, _, labels = name.partition('{')
                    labels = dict(re.findall(r'(\w+)="([^"]*)"', labels))
                    if not key.endswith('_total') or labels.get('tp_rank', '0') != '0':
                        continue
                    for label in ('cache_source', 'pool'):
                        if label in labels:
                            key += f'{{{label}="{labels[label]}"}}'
                    totals[key] = totals.get(key, 0.0) + float(value)
        return totals

    def turn(self, messages):
        """Stream one chat turn; return TTFT, the assistant message to send back, and usage."""
        body = {'model': self.model, 'messages': messages, 'stream': True, 'max_tokens': 256,
                'reasoning_effort': 'low', 'stream_options': {'include_usage': True}}
        req = urllib.request.Request(self.base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers=self.headers)
        started = time.monotonic()
        ttft, content, reasoning, usage = None, [], [], {}
        with OPENER.open(req, timeout=1800) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:') or line == 'data: [DONE]':
                    continue
                chunk = json.loads(line[5:])
                usage = chunk.get('usage') or usage
                for choice in chunk.get('choices') or []:
                    delta = choice.get('delta') or {}
                    if ttft is None and (delta.get('content') or delta.get('reasoning_content')):
                        ttft = time.monotonic() - started
                    content.append(delta.get('content') or '')
                    reasoning.append(delta.get('reasoning_content') or '')
        reply = {'role': 'assistant', 'content': ''.join(content)}
        if any(reasoning):
            reply['reasoning_content'] = ''.join(reasoning)
        return ttft, reply, usage


def document(rng, sentences):
    return ' '.join(f'In {rng.randint(1780, 1990)}, town {rng.randint(1, 400)} recorded its '
                    f'{rng.choice(WORDS)} with {rng.randint(2, 9000)} visitors.' for _ in range(sentences))


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('label')
    parser.add_argument('--sessions', type=int, default=96)
    parser.add_argument('--turns', type=int, default=3)
    parser.add_argument('--doc-tokens', type=int, default=24000)
    parser.add_argument('--concurrency', type=int, default=8)
    args = parser.parse_args()
    client = Client()
    # Size documents from one measured sample; distinct seeds keep sessions from sharing prefixes.
    _, _, usage = client.turn([{'role': 'user', 'content': document(random.Random(-1), 500)}])
    per_sentence = usage['prompt_tokens'] / 500
    sentences = int(args.doc_tokens / per_sentence)
    sessions = [[{'role': 'user', 'content': f'Session {index}. Records:\n' + document(random.Random(index), sentences)
                  + '\n\n' + QUESTIONS[0]}] for index in range(args.sessions)]
    before = client.metrics()
    turns = []
    started = time.monotonic()
    for turn in range(args.turns):
        if turn:
            for messages in sessions:
                messages.append({'role': 'user', 'content': QUESTIONS[turn % len(QUESTIONS)]})
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(client.turn, [list(messages) for messages in sessions]))
        for messages, (_, reply, _) in zip(sessions, results):
            messages.append(reply)
        ttfts = [ttft for ttft, _, _ in results if ttft is not None]
        prompts = [usage.get('prompt_tokens', 0) for _, _, usage in results]
        turns.append({'turn': turn + 1, 'requests': len(results), 'ttft_mean_s': statistics.mean(ttfts),
                      'ttft_p50_s': percentile(ttfts, 0.5), 'ttft_p90_s': percentile(ttfts, 0.9),
                      'prompt_tokens_mean': statistics.mean(prompts)})
        print(json.dumps(turns[-1]), flush=True)
    elapsed = time.monotonic() - started
    after = client.metrics()
    counters = {key: after[key] - before.get(key, 0.0) for key in sorted(after)
                if any(word in key for word in ('cache', 'load_back', 'prompt_tokens', 'generation_tokens'))
                and after[key] != before.get(key, 0.0)}
    result = {'label': args.label, 'sessions': args.sessions, 'turns_per_session': args.turns,
              'doc_tokens': args.doc_tokens, 'concurrency': args.concurrency, 'elapsed_s': round(elapsed, 1),
              'per_turn': turns, 'counter_deltas': counters}
    out = ROOT / '.runs/bench' / f'{args.label}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'elapsed_s': result['elapsed_s'], 'counter_deltas': counters}, indent=1))
    print(f'({out})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
