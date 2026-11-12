#!/usr/bin/env python3
"""How often the model miscopies text it just read: Claude Code's Read-then-Edit pattern, replayed.

Each trial shows a source file as a Read tool result (Claude Code's `cat -n` format) and asks for one
Edit whose old_string is a given line range. A trial fails when old_string is not an exact substring
of the file: what Claude Code reports as "String to replace not found". Conditions vary only the
request's sampling fields, so the same server can be compared with itself. Stdlib only; on c04.

    edit_copy_check.py [--trials 48] [--concurrency 16] [--effort low]"""
import argparse
import concurrent.futures
import datetime
import difflib
import json
from pathlib import Path
import random
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
SOURCES = ['smoke_test.py', 'web_search.py']
CONDITIONS = {'default (T=1, top_p=1)': {}, 'top_p 0.95': {'top_p': 0.95}, 'temperature 0': {'temperature': 0.0}}
EDIT = {'name': 'Edit', 'description': 'Performs exact string replacements in files. old_string must match the '
        'file exactly, including indentation, and must not include the line-number prefix of Read output.',
        'input_schema': {'type': 'object', 'required': ['file_path', 'old_string', 'new_string'], 'properties': {
            'file_path': {'type': 'string'}, 'old_string': {'type': 'string'}, 'new_string': {'type': 'string'}}}}
READ = {'name': 'Read', 'description': 'Reads a file; output is cat -n style, line number + tab + line.',
        'input_schema': {'type': 'object', 'required': ['file_path'], 'properties': {'file_path': {'type': 'string'}}}}


def trial(cfg, key, source, start, end, sampling, effort):
    path = f'/work/{source}'
    text = (ROOT / source).read_text()
    shown = ''.join(f'{number:>6}\t{line}\n' for number, line in enumerate(text.splitlines(), 1))
    body = {'model': cfg['served_model_name'], 'max_tokens': 8192, 'tools': [READ, EDIT],
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': effort}, **sampling,
            'messages': [
                {'role': 'user', 'content': f'Read {path}.'},
                {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_read', 'name': 'Read',
                                                   'input': {'file_path': path}}]},
                {'role': 'user', 'content': [
                    {'type': 'tool_result', 'tool_use_id': 'toolu_read', 'content': shown},
                    {'type': 'text', 'text': f'Now use one Edit call on {path}: old_string is exactly lines '
                                             f'{start} to {end}, and new_string is the same lines with the '
                                             'comment "  # reviewed" appended to the last of them.'}]}]}
    request = urllib.request.Request(f'http://{cfg["bind_host"]}:{cfg["port"]}/v1/messages', method='POST',
                                     data=json.dumps(body).encode(), headers={
                                         'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    try:
        with OPENER.open(request, timeout=900) as response:
            reply = json.load(response)
    except (urllib.error.URLError, OSError) as error:
        return {'outcome': 'request error', 'detail': str(error)[:200]}
    edits = [block['input'] for block in reply['content'] if block['type'] == 'tool_use' and block['name'] == 'Edit']
    if not edits:
        return {'outcome': 'no edit', 'stop': reply.get('stop_reason')}
    old = edits[0].get('old_string', '')
    expected = '\n'.join(text.splitlines()[start - 1:end])
    if old and old in text:
        return {'outcome': 'exact' if old.strip('\n') == expected else 'found, other range'}
    diff = [line for line in difflib.unified_diff(expected.splitlines(), old.splitlines(), lineterm='', n=0)
            if line[:1] in '+-' and line[:3] not in ('+++', '---')]
    return {'outcome': 'miscopied', 'diff': diff[:6], 'source': source, 'lines': [start, end]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trials', type=int, default=48, help='per condition')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--effort', default='low', choices=['low', 'high', 'max'])
    parser.add_argument('--only', nargs='+', choices=list(CONDITIONS))
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'serving.json').read_text())
    key = (ROOT / '.secrets/api-key').read_text().strip()
    rng = random.Random(0)
    cases = []
    for _ in range(args.trials):
        source = rng.choice(SOURCES)
        count = len((ROOT / source).read_text().splitlines())
        start = rng.randint(1, count - 8)
        cases.append((source, start, start + rng.randint(2, 7)))
    results = {}
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for name in args.only or CONDITIONS:
            runs = list(pool.map(lambda case: trial(cfg, key, *case, CONDITIONS[name], args.effort), cases))
            counts = {}
            for run in runs:
                counts[run['outcome']] = counts.get(run['outcome'], 0) + 1
            results[name] = {'counts': counts, 'miscopies': [run for run in runs if run['outcome'] == 'miscopied']}
            print(f'{name}: {counts}', flush=True)
            for run in results[name]['miscopies'][:4]:
                print(f'   {run["source"]}:{run["lines"]} {json.dumps(run["diff"], ensure_ascii=False)[:300]}')
    out = ROOT / '.runs/edit-copy' / f'{datetime.datetime.now():%Y%m%d-%H%M%S}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    weights = 'abliterated' if cfg.get('checkpoint_overlay') else 'official'
    out.write_text(json.dumps({'weights': weights, 'speculative': cfg.get('speculative'), 'effort': args.effort,
                               'trials': args.trials, 'results': results}, indent=1, ensure_ascii=False) + '\n')
    print(f'weights {weights}; results in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
