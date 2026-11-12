#!/usr/bin/env python3
"""Corruption in long generated files: an OBJ sphere and an SVG, written through a Write tool call.

Each trial streams /v1/messages as Claude Code does and asks for one long file whose every line can be
checked. OBJ: each `v` must lie on the unit sphere and each `f` must index existing vertices. SVG: it
must parse as XML. In both, any non-ASCII character, U+FFFD, control character or leaked special
token counts as a stray character. Conditions vary only the request's sampling fields. Stdlib only.

    long_output_check.py [--trials 6] [--concurrency 12] [--effort low] [--only ...]"""
import argparse
import concurrent.futures
import datetime
import json
import math
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# Explicit fields, so a server-side default changes only 'client default' (what Claude Code sends).
CONDITIONS = {'T=1, top_p=1': {'temperature': 1.0, 'top_p': 1.0}, 'top_p 0.95': {'temperature': 1.0, 'top_p': 0.95},
              'temperature 0': {'temperature': 0.0}, 'client default': {}}
WRITE = {'name': 'Write', 'description': 'Writes a file to the local filesystem, overwriting it.',
         'input_schema': {'type': 'object', 'required': ['file_path', 'content'], 'properties': {
             'file_path': {'type': 'string'}, 'content': {'type': 'string'}}}}
TASKS = {
    'obj': ('Use the Write tool once to create /work/sphere.obj: a UV sphere of radius 1 centred at the origin, '
            '24 segments and 16 rings. Write every vertex as "v x y z" with 4 decimals (north pole, then the '
            '15 inner rings of 24 vertices, then the south pole), then every face as "f a b c" or "f a b c d" '
            'with 1-based indices. Only v and f lines, no comments. Write out the whole file; do not abbreviate.'),
    'svg': ('Use the Write tool once to create /work/skyline.svg: a detailed night city skyline, 1200x600, '
            'with at least 150 shapes (buildings, lit and dark windows, stars, a moon). ASCII only, no <text> '
            'elements, no scripts. Write out the whole file; do not abbreviate.'),
}
SPECIAL = re.compile(r'<\|[a-z_]+\|>|\[EOS\]|</?think>')
STRAY = re.compile(r'[^\x09\x0a\x0d\x20-\x7e]')


def stream(cfg, key, body):
    request = urllib.request.Request(f'http://{cfg["bind_host"]}:{cfg["port"]}/v1/messages', method='POST',
                                     data=json.dumps(dict(body, stream=True)).encode(), headers={
                                         'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    with OPENER.open(request, timeout=1800) as response:
        raw = response.read()
    partial, stop, usage = '', None, {}
    for line in raw.decode('utf-8').splitlines():
        if line.startswith('data:'):
            event = json.loads(line[5:])
            if event['type'] == 'content_block_delta':
                partial += event['delta'].get('partial_json', '')
            elif event['type'] == 'message_delta':
                stop, usage = event['delta'].get('stop_reason'), event.get('usage') or {}
    return partial, stop, usage


def check_obj(content):
    lines = [line for line in content.splitlines() if line.strip()]
    vertices, bad = 0, []
    for number, line in enumerate(lines, 1):
        parts = line.split()
        try:
            if parts[0] == 'v' and len(parts) == 4:
                x, y, z = map(float, parts[1:])
                vertices += 1
                if abs(math.sqrt(x * x + y * y + z * z) - 1) > 0.01:
                    bad.append(f'{number}: off-sphere {line!r}')
            elif parts[0] == 'f' and len(parts) in (4, 5):
                if any(not 1 <= int(index.split('/')[0]) <= 362 for index in parts[1:]):
                    bad.append(f'{number}: index out of range {line!r}')
            else:
                bad.append(f'{number}: malformed {line!r}')
        except ValueError:
            bad.append(f'{number}: unparsable {line!r}')
    if vertices != 362:
        bad.append(f'{vertices} vertices, expected 362')
    return bad


def check_svg(content):
    try:
        root = ET.fromstring(content)
    except ET.ParseError as error:
        return [f'XML error: {error}']
    shapes = sum(1 for element in root.iter() if element.tag.split('}')[-1] in (
        'rect', 'circle', 'ellipse', 'polygon', 'path', 'line', 'polyline'))
    return [] if shapes >= 150 else [f'only {shapes} shapes']


def trial(cfg, key, task, sampling, effort):
    body = {'model': cfg['served_model_name'], 'max_tokens': 32000, 'tools': [WRITE],
            'thinking': {'type': 'adaptive'}, 'output_config': {'effort': effort}, **sampling,
            'messages': [{'role': 'user', 'content': TASKS[task]}]}
    try:
        partial, stop, usage = stream(cfg, key, body)
    except (urllib.error.URLError, OSError, ValueError) as error:
        return {'task': task, 'outcome': 'request error', 'detail': str(error)[:200]}
    try:
        content = json.loads(partial).get('content', '')
    except ValueError as error:
        return {'task': task, 'outcome': 'tool JSON broken', 'detail': str(error), 'stop': stop}
    stray = [{'char': repr(match.group()), 'context': content[max(0, match.start() - 40):match.end() + 40]}
             for pattern in (STRAY, SPECIAL) for match in pattern.finditer(content)]
    problems = (check_obj if task == 'obj' else check_svg)(content)
    outcome = 'clean' if not problems and not stray else ('stray characters' if stray else 'invalid file')
    # A minus sign inside a face line is the decode-drift symptom; OBJ faces here never need one.
    face_minus = sum(line.startswith('f ') and '-' in line for line in content.splitlines())
    return {'task': task, 'outcome': outcome, 'stop': stop, 'output_tokens': usage.get('output_tokens'),
            'face_minus': face_minus,
            'stray': stray[:5], 'stray_count': len(stray), 'problems': problems[:5], 'problem_count': len(problems)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trials', type=int, default=6, help='per task and condition')
    parser.add_argument('--concurrency', type=int, default=12)
    parser.add_argument('--effort', default='low', choices=['low', 'high', 'max'])
    parser.add_argument('--only', nargs='+', choices=list(CONDITIONS))
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'serving.json').read_text())
    key = (ROOT / '.secrets/api-key').read_text().strip()
    results = {}
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        for name in args.only or CONDITIONS:
            cases = [task for task in TASKS for _ in range(args.trials)]
            runs = list(pool.map(lambda task: trial(cfg, key, task, CONDITIONS[name], args.effort), cases))
            results[name] = runs
            for task in TASKS:
                mine = [run for run in runs if run['task'] == task]
                counts = {}
                for run in mine:
                    counts[run['outcome']] = counts.get(run['outcome'], 0) + 1
                tokens = sorted(run.get('output_tokens') or 0 for run in mine)
                minus = sum(run.get('face_minus', 0) for run in mine)
                print(f'{name} / {task}: {counts}; output tokens {tokens}'
                      + (f'; face lines with "-": {minus}' if task == 'obj' else ''), flush=True)
                for run in mine:
                    for item in run.get('stray', [])[:2]:
                        print(f'   stray {item["char"]}: {item["context"]!r}')
                    for problem in run.get('problems', [])[:2]:
                        print(f'   {problem[:160]}')
    out = ROOT / '.runs/long-output' / f'{datetime.datetime.now():%Y%m%d-%H%M%S}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    weights = 'abliterated' if cfg.get('checkpoint_overlay') else 'official'
    out.write_text(json.dumps({'weights': weights, 'speculative': cfg.get('speculative'), 'effort': args.effort,
                               'results': results}, indent=1, ensure_ascii=False) + '\n')
    print(f'weights {weights}; results in {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
