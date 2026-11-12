#!/usr/bin/env python3
"""Record a whole Claude Code session against the running endpoint, to diagnose tool-use failures.

Unlike claude_code_check.py, the relay keeps each request body and streamed response (never the
headers, so no key) under .runs/claude-code-trace/<ts>/, so run it on synthetic tasks only. The
task edits a small fixture project with tabs, CJK comments and non-ASCII strings; the report
lists every tool call, each failed edit beside the file text it targeted, and any U+FFFD,
control characters or leaked special tokens in model output.

    claude_code_trace.py [--runs 1]"""
import argparse
import datetime
import json
from pathlib import Path
import re
import sys
import threading

import claude_code_check as check

FIXTURE = {
    'inventory.py': '''"""Inventory helpers."""

# 库存数量的默认上限
MAX_STOCK = 500


class Item:
    def __init__(self, name, price, qty):
        self.name = name  # 商品名称，例如 "café crème"
        self.price = price
        self.qty = qty


def calc_total(items):
    # 计算总价 — 不含折扣
    total = 0
    for item in items:
        total += item.price * item.qty
    return total


def restock(item, amount):
    # 补货，但不能超过上限
    item.qty = min(item.qty + amount, MAX_STOCK)
    return item
''',
    'legacy.py': '''from inventory import calc_total


def discounted_total(items, rate):
\t"""Apply a discount rate (0.1 = 10 %) to the order total."""
\ttotal = calc_total(items)
\tif rate > 0:
\t\ttotal = total * (1 - rate)
\t\t# apply the discount → final price
\t\ttotal = total * (1 - rate)
\treturn round(total, 2)
''',
    'report.py': '''import sys

from inventory import Item, calc_total


def main(argv):
    items = [Item("café crème", 3.5, 4), Item("thé vert", 2.25, 10), Item("📦 box", 1.0, 3)]
    print(f"Total → {calc_total(items):.2f} €")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
''',
    'tests/test_inventory.py': '''from inventory import Item, calc_total, restock


def test_calc_total():
    assert calc_total([Item("a", 2.0, 3), Item("b", 1.5, 2)]) == 9.0


def test_restock_caps():
    assert restock(Item("a", 1.0, 490), 50).qty == 500
''',
}
TASK = ('In this project: (1) rename the function calc_total to compute_total everywhere: the definition, '
        'all imports and calls, and the tests. (2) legacy.py applies the discount twice; fix that bug. '
        '(3) Translate the Chinese comments in inventory.py into English. (4) Add a --json flag to '
        'report.py\'s main that prints the items and the total as JSON instead. Do not run any commands; '
        'just edit the files. Finish with a one-paragraph summary.')
TOOLS = ['Read', 'Edit', 'Write', 'Glob', 'Grep']
SPECIAL = re.compile(r'<\|[a-z_]+\|>|\[EOS\]|<think>|</think>|<\|close\|>|<\|sep\|>')
CONTROL = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f�]')


def full_request(path, headers, raw):
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = {'unparsed': raw.decode(errors='replace')}
    return {'path': path, 'body': body}


def full_response(status, content_type, raw):
    return {'status': status, 'content_type': content_type, 'text': raw.decode('utf-8', errors='replace'),
            'invalid_utf8': raw != raw.decode('utf-8', errors='replace').encode()}


def sse_blocks(text):
    """Rebuild the content blocks of a streamed /v1/messages response."""
    blocks, stop = {}, None
    for line in text.splitlines():
        if not line.startswith('data:'):
            continue
        try:
            event = json.loads(line[5:])
        except ValueError:
            continue
        kind = event.get('type')
        if kind == 'content_block_start':
            block = dict(event['content_block'])
            block.setdefault('partial', '')
            blocks[event['index']] = block
        elif kind == 'content_block_delta':
            delta, block = event['delta'], blocks.setdefault(event['index'], {'partial': ''})
            for key in ('text', 'thinking', 'partial_json'):
                if key in delta:
                    block[key if key != 'partial_json' else 'partial'] = block.get(
                        key if key != 'partial_json' else 'partial', '') + delta[key]
        elif kind == 'message_delta':
            stop = event['delta'].get('stop_reason')
    for block in blocks.values():
        if block.get('type') == 'tool_use' and block.get('partial'):
            try:
                block['input'] = json.loads(block['partial'])
            except ValueError as error:
                block['input_error'] = str(error)
    return [blocks[i] for i in sorted(blocks)], stop


def analyze(records):
    """Pair each tool_use with its tool_result (sent in the next request) and flag problems."""
    calls, results, oddities = {}, {}, []
    for number, record in enumerate(records):
        request, response = record['request'], record.get('response', {})
        if request['path'].split('?')[0] != '/v1/messages':
            continue
        for message in request['body'].get('messages', []):
            for block in message.get('content') if isinstance(message.get('content'), list) else []:
                if block.get('type') == 'tool_result':
                    content = block.get('content')
                    text = content if isinstance(content, str) else ' '.join(
                        part.get('text', '') for part in content or [] if isinstance(part, dict))
                    results[block['tool_use_id']] = {'is_error': block.get('is_error', False), 'text': text}
        if 'text/event-stream' not in response.get('content_type', ''):
            continue
        blocks, stop = sse_blocks(response['text'])
        if response.get('invalid_utf8'):
            oddities.append({'request': number, 'what': 'response bytes are not valid UTF-8'})
        for block in blocks:
            for field in ('text', 'thinking', 'partial'):
                value = block.get(field) or ''
                for pattern, label in ((SPECIAL, 'special token'), (CONTROL, 'control/U+FFFD')):
                    for match in pattern.finditer(value):
                        oddities.append({'request': number, 'block': block.get('type'), 'field': field,
                                         'what': label, 'match': repr(match.group()),
                                         'context': value[max(0, match.start() - 60):match.end() + 60]})
            if block.get('type') == 'tool_use':
                calls[block['id']] = {'request': number, 'name': block.get('name'), 'input': block.get('input'),
                                      'input_error': block.get('input_error'), 'stop': stop}
    report = []
    for call_id, call in calls.items():
        result = results.get(call_id, {})
        row = {'request': call['request'], 'tool': call['name'], 'error': result.get('is_error'),
               'result': (result.get('text') or '')[:300]}
        if call.get('input_error'):
            row['input_error'] = call['input_error']
        if call['name'] in ('Edit', 'MultiEdit') and result.get('is_error'):
            row['input'] = call['input']
        report.append(row)
    return report, oddities


def fixture(directory):
    for name, text in FIXTURE.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads((check.ROOT / 'serving.json').read_text())
    model, key = cfg['served_model_name'], (check.ROOT / '.secrets/api-key').read_text().strip()
    check.request_fields, check.response_fields = full_request, full_response
    relay = check.Relay(f'http://{cfg["bind_host"]}:{cfg["port"]}')
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    out = check.ROOT / '.runs/claude-code-trace' / f'{datetime.datetime.now():%Y%m%d-%H%M%S}'
    summary = []
    for run in range(args.runs):
        workdir = out / f'run{run}'
        fixture(workdir / 'cwd')
        result = check.run_claude(workdir, relay, model, key, TASK, TOOLS, timeout=1800)
        records = result.pop('requests')
        (workdir / 'records.json').write_text(json.dumps(records, indent=1, ensure_ascii=False))
        report, oddities = analyze(records)
        failed = [row for row in report if row['error']]
        tools = {}
        for row in report:
            tools[row['tool']] = tools.get(row['tool'], 0) + 1
        line = {'run': run, 'exit': result['exit_code'], 'requests': len(records), 'tool_calls': tools,
                'failed_calls': len(failed), 'oddities': len(oddities), 'result': result['result'][:300]}
        summary.append(line)
        (workdir / 'report.json').write_text(json.dumps({'summary': line, 'calls': report, 'oddities': oddities},
                                                        indent=1, ensure_ascii=False))
        print(json.dumps(line, ensure_ascii=False), flush=True)
    relay.shutdown()
    print(f'traces under {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
