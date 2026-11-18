#!/usr/bin/env python3
"""Anthropic /v1/messages checks against the running endpoint, run on c04 (mirrors ds41ctl's checks).

  vision, tool, stream   image input, a tool call and its result, each as JSON and streamed
  effort                 every Anthropic output_config.effort and OpenAI reasoning_effort level answers
  search                 OpenSERP web search through web_search.py: the answer cites the evidence URLs
  no-search              a request without the server search tool makes no provider call
  budget                 thinking.budget_tokens: reasoning stops at the budget and an answer follows;
                         raw token ids through OpenAI chat custom_params, mixed budgets in one batch,
                         and a budgeted tool call round trip
  speed                  decode rate of unbudgeted streamed requests (compare before and after a change)
  documents              a two-page PDF (text page, image-only page) as an attachment (JSON) and inside a
                         Read tool result (streamed); check_documents.py is the full regression

Requests carry no sampling overrides, except `documents` (temperature 0, as check_documents.py). `budget`
loads the tokenizer and `documents` needs Pillow, so run those with the venv's Python.
Results go to .runs/checks/anthropic-<time>.json.

    anthropic_check.py [--only vision tool stream effort search no-search budget speed documents]"""
import argparse
import base64
import concurrent.futures
import datetime
import json
from pathlib import Path
import re
import secrets
import statistics
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit

from smoke_test import png

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHECKS = ['vision', 'tool', 'stream', 'effort', 'search', 'no-search', 'budget', 'speed', 'documents']
LOW = {'output_config': {'effort': 'low'}}
TOOLS = [{'name': 'fixture_lookup', 'description': 'Return the deployment test value for a key.',
          'input_schema': {'type': 'object', 'properties': {'key': {'type': 'string'}}, 'required': ['key']}}]
LOOKUP = 'Call fixture_lookup with key deployment, then report exactly the value the tool returns.'
SUDOKU = ('Solve this Sudoku without tools. Zero means blank. Check every row, column, and 3x3 box. '
          'Return the nine completed rows.\n100007090\n030020008\n009600500\n005300900\n'
          '010080002\n600004000\n300000010\n040000007\n007000300')


class Client:
    def __init__(self):
        self.cfg = json.loads((ROOT / 'serving.json').read_text())
        self.base = f'http://{self.cfg["bind_host"]}:{self.cfg["port"]}'
        self.model = self.cfg['served_model_name']
        self.headers = {'Authorization': 'Bearer ' + (ROOT / '.secrets/api-key').read_text().strip(),
                        'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'}

    def post(self, path, body, timeout=900):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=self.headers)
        try:
            return OPENER.open(req, timeout=timeout)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read(1500).decode(errors="replace")}') from error

    def messages(self, **body):
        started = time.monotonic()
        with self.post('/v1/messages', dict(model=self.model, **body)) as response:
            result = json.load(response)
        result['elapsed_s'] = round(time.monotonic() - started, 2)
        return result

    def stream(self, **body):
        """Stream a message; rebuild its content blocks and record when output tokens arrived."""
        blocks, partial, events, stamps = {}, {}, [], []
        started = time.monotonic()
        with self.post('/v1/messages', dict(model=self.model, stream=True, **body)) as response:
            for line in response:
                if not line.startswith(b'data:') or not line[5:].strip():
                    continue
                event = json.loads(line[5:])
                events.append(event)
                kind, index = event.get('type'), event.get('index', 0)
                if kind == 'error':
                    raise RuntimeError('stream error: ' + json.dumps(event))
                if kind == 'content_block_start':
                    blocks[index] = dict(event['content_block'])
                elif kind == 'content_block_delta':
                    stamps.append(time.monotonic() - started)
                    delta = event['delta']
                    block = blocks.setdefault(index, {})
                    if delta.get('type') == 'text_delta':
                        block['text'] = block.get('text', '') + delta['text']
                    elif delta.get('type') == 'thinking_delta':
                        block['thinking'] = block.get('thinking', '') + delta['thinking']
                    elif delta.get('type') == 'input_json_delta':
                        partial[index] = partial.get(index, '') + delta['partial_json']
        for index, raw in partial.items():
            blocks[index]['input'] = json.loads(raw or '{}')
        final = [e for e in events if e.get('type') == 'message_delta']
        return {'content': [blocks[i] for i in sorted(blocks)],
                'stop_reason': final[-1]['delta'].get('stop_reason') if final else None,
                'usage': final[-1].get('usage') if final else None,
                'stopped': any(e.get('type') == 'message_stop' for e in events), 'stamps': stamps,
                'elapsed_s': round(time.monotonic() - started, 2)}


def text(response):
    return ''.join(block.get('text', '') for block in response['content'] if block.get('type') == 'text')


def thinking(response):
    return ''.join(block.get('thinking', '') for block in response['content'] if block.get('type') == 'thinking')


def brief(response, **extra):
    return {'stop_reason': response.get('stop_reason'), 'usage': response.get('usage'),
            'thinking_chars': len(thinking(response)), 'text': text(response)[:300],
            'elapsed_s': response.get('elapsed_s'), **extra}


def check_vision(client):
    image = base64.b64encode(png(256, 256, lambda x, y: (220, 20, 20) if x < 128 else (20, 40, 220))).decode()
    body = dict(max_tokens=4096, messages=[{'role': 'user', 'content': [
        {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image}},
        {'type': 'text', 'text': 'Answer exactly in the form "left: <color>, right: <color>".'}]}], **LOW)
    results = {}
    for mode, call in (('json', client.messages), ('stream', client.stream)):
        response = call(**body)
        left, _, right = text(response).lower().partition('right')
        results[mode] = ('red' in left and 'blue' in right, brief(response))
    return all(ok for ok, _ in results.values()), {mode: detail for mode, (_, detail) in results.items()}


def tool_round_trip(client, call, **options):
    messages = [{'role': 'user', 'content': LOOKUP}]
    first = call(max_tokens=4096, messages=messages, tools=TOOLS,
                 tool_choice={'type': 'tool', 'name': 'fixture_lookup'}, **options)
    calls = [block for block in first['content'] if block.get('type') == 'tool_use']
    if len(calls) != 1 or calls[0].get('input', {}).get('key') != 'deployment':
        return False, {'first': brief(first, tool_calls=calls)}
    value = 'KIMI_OK_' + secrets.token_hex(4)
    messages += [{'role': 'assistant', 'content': first['content']},
                 {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': calls[0]['id'], 'content': value}]}]
    second = call(max_tokens=4096, messages=messages, tools=TOOLS, **options)
    return value in text(second), {'first': brief(first, tool_calls=calls), 'second': brief(second, expected=value)}


def check_tool(client):
    return tool_round_trip(client, client.messages, **LOW)


def check_stream(client):
    return tool_round_trip(client, client.stream, **LOW)


def check_effort(client):
    detail = {}
    for effort in ('low', 'medium', 'high', 'xhigh', 'max'):
        try:
            response = client.messages(max_tokens=1024, thinking={'type': 'adaptive'}, output_config={'effort': effort},
                                       messages=[{'role': 'user', 'content': 'Reply with ok.'}])
            detail[f'anthropic_{effort}'] = response.get('stop_reason')
        except RuntimeError as error:
            detail[f'anthropic_{effort}'] = str(error)[:120]
        try:
            with client.post('/v1/chat/completions', {'model': client.model, 'max_tokens': 1024, 'reasoning_effort': effort,
                                                     'messages': [{'role': 'user', 'content': 'Reply with ok.'}]}) as r:
                detail[f'openai_{effort}'] = json.load(r)['choices'][0]['finish_reason']
        except RuntimeError as error:
            detail[f'openai_{effort}'] = str(error)[:120]
    return all(value in ('end_turn', 'stop') for value in detail.values()), detail


def citations_allowed(answer, domains):
    hosts = []
    for url in re.findall(r'\]\((https?://[^)\s]+)\)', answer):
        try:
            host = (urlsplit(url).hostname or '').lower().rstrip('.')
        except ValueError:
            continue
        if host:
            hosts.append(host)
    return bool(hosts) and all(any(host == domain or host.endswith('.' + domain)
                                   for domain in domains) for host in hosts)


def search_log_lines():
    log = ROOT / 'sglang.log'
    return log.read_text(errors='replace').count('OpenSERP web search attempt') if log.exists() else 0


def check_search(client):
    results = {}
    domains = ('github.com', 'docs.sglang.ai', 'docs.sglang.io', 'lmsys.org')
    for mode, version in (('json', 'web_search_20250305'), ('stream', 'web_search_20260209')):
        call = client.stream if mode == 'stream' else client.messages
        response = call(max_tokens=4096, system='Answer in one sentence and cite one source URL from the search evidence.',
                        messages=[{'role': 'user', 'content': 'Perform a web search for the query: SGLang Anthropic Messages API support'}],
                        tools=[{'type': version, 'name': 'web_search', 'allowed_domains': list(domains)}],
                        tool_choice={'type': 'tool', 'name': 'web_search'}, **LOW)
        answer = text(response)
        results[mode] = (citations_allowed(answer, domains), brief(response, tool_type=version))
    return all(ok for ok, _ in results.values()), {mode: detail for mode, (_, detail) in results.items()}


def check_no_search(client):
    # A client-defined tool that happens to be called web_search is not Anthropic's server tool.
    before = search_log_lines()
    response = client.messages(max_tokens=2048, messages=[{'role': 'user', 'content': 'Reply with the word ready.'}],
                               tools=[{'name': 'web_search', 'description': 'A client tool.',
                                       'input_schema': {'type': 'object', 'properties': {}}}], **LOW)
    after = search_log_lines()
    return after == before and response.get('stop_reason') in ('end_turn', 'tool_use'), \
        brief(response, search_log_lines_before=before, search_log_lines_after=after)


def check_budget(client):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(client.cfg['model_path'], trust_remote_code=True, local_files_only=True)
    close = tokenizer.encode('<|close|>think<|sep|>', add_special_tokens=False)
    detail, ok = {'think_close_ids': close}, True
    # Raw ids: where the forced close marker lands relative to the budget.
    for budget in (16, 128):
        with client.post('/v1/chat/completions', {
                'model': client.model, 'messages': [{'role': 'user', 'content': SUDOKU}], 'max_tokens': 4096,
                'reasoning_effort': 'max', 'custom_params': {'thinking_budget': budget}, 'return_token_ids': True}) as r:
            response = json.load(r)
        choice = response['choices'][0]
        ids = choice.get('response_token_ids') or []
        boundary = next((i for i in range(len(ids)) if ids[i:i + len(close)] == close), None)
        passed = boundary is not None and boundary <= budget and bool(choice['message'].get('content'))
        ok &= passed
        detail[f'raw_{budget}'] = {'passed': passed, 'boundary': boundary, 'finish_reason': choice['finish_reason'],
                                   'completion_tokens': response['usage']['completion_tokens'],
                                   'content': (choice['message'].get('content') or '')[:120]}

    def probe(item):
        budget, mode = item
        call = client.stream if mode == 'stream' else client.messages
        response = call(max_tokens=8192, messages=[{'role': 'user', 'content': SUDOKU}],
                        thinking={'type': 'enabled', 'budget_tokens': budget})
        count = len(tokenizer.encode(thinking(response), add_special_tokens=False))
        ended = response.get('stopped') if mode == 'stream' else response.get('stop_reason') == 'end_turn'
        # Re-tokenized reasoning can differ from the generated ids by a few tokens.
        return f'anthropic_{budget}_{mode}', 0 < count <= budget + 8 and bool(text(response)) and ended, \
            brief(response, retokenized_thinking_tokens=count, budget=budget)
    # Mixed budgets in one batch: isolation between requests and draft rollback under the mask.
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for name, passed, info in pool.map(probe, [(1024, 'json'), (1024, 'stream'), (2048, 'json'), (2048, 'stream')]):
            ok &= passed
            detail[name] = dict(info, passed=passed)
    passed, info = tool_round_trip(client, client.stream, thinking={'type': 'enabled', 'budget_tokens': 1024})
    ok &= passed
    detail['budget_tool'] = dict(info, passed=passed)
    return ok, detail


def check_speed(client, runs=8):
    rates = []
    for _ in range(runs):
        response = client.stream(max_tokens=1024, messages=[{'role': 'user', 'content':
                                 'Explain how a ferry crossing schedule is planned, in detail.'}])
        stamps, tokens = response['stamps'], (response['usage'] or {}).get('output_tokens', 0)
        if len(stamps) > 1 and tokens > 1:
            rates.append((tokens - 1) / (stamps[-1] - stamps[0]))
    return bool(rates), {'decode_tokens_per_s': [round(r, 1) for r in rates],
                         'median': round(statistics.median(rates), 1) if rates else None}


def check_documents(client):
    from check_documents import read_pdf
    from test_documents import fixture_pdf
    code, visual = 'TEXT_' + secrets.token_hex(3), 'VISUAL_' + secrets.token_hex(3)
    pdf, detail, ok = fixture_pdf(code, visual), {}, True
    for name, nested, stream in (('attachment_json', False, False), ('tool_result_stream', True, True)):
        passed, response = read_pdf(client, pdf, code, visual, nested, stream)
        ok &= passed
        detail[name] = dict(brief(response), passed=passed)
    return ok, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='+', choices=CHECKS, default=CHECKS)
    args = parser.parse_args()
    client = Client()
    runs = {'vision': check_vision, 'tool': check_tool, 'stream': check_stream, 'effort': check_effort,
            'search': check_search,
            'no-search': check_no_search, 'budget': check_budget, 'speed': check_speed,
            'documents': check_documents}
    results = {}
    for name in args.only:
        try:
            ok, detail = runs[name](client)
        except Exception as exc:
            ok, detail = False, {'error': f'{type(exc).__name__}: {exc}'}
        results[name] = {'ok': ok, 'detail': detail}
        print(f'{"ok  " if ok else "FAIL"} {name}: {json.dumps(detail, ensure_ascii=False)[:1500]}', flush=True)
    out = ROOT / '.runs/checks' / f'anthropic-{datetime.datetime.now():%Y%m%d-%H%M%S}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    failed = [name for name, result in results.items() if not result['ok']]
    print(f'\n{len(failed)} failed: {", ".join(failed)}' if failed else '\nall checks passed', f'({out})')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
