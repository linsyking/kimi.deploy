#!/usr/bin/env python3
"""Drive the Claude Code CLI against the running endpoint, run on c04 (stdlib only).

A loopback relay between claude and the server records, per request, only the fields that
decide limits and tools (max_tokens, thinking, output_config, tool types, document count, stream, auth scheme)
and each response's status, stop reason and usage. Prompts, model output and keys are not stored,
except a flag for whether a reply contained a URL. claude runs with HOME and CLAUDE_CONFIG_DIR in
a fresh directory under .runs/claude-code, so the user's settings, hooks, plugins and history are
neither used nor changed.

  basic        a one-line answer, no tools
  max-tokens   CLAUDE_CODE_MAX_OUTPUT_TOKENS=256 on a long answer: requests carry 256, responses stop there
  search       Claude Code's WebSearch tool: its server-tool subrequest comes back with OpenSERP source URLs
  thinking     MAX_THINKING_TOKENS=2048: records what Claude Code sends for thinking with this model
  pdf          the Read tool on a two-page PDF (a text code; an image-only code and a bar chart): Claude Code
               sends it as a document block and the answer has both codes (fixture: run with the venv's Python)

    claude_code_check.py [--only basic max-tokens search thinking pdf]"""
import argparse
import datetime
import http.server
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
CLAUDE = Path('/scratch_local/user_data/yiming/.local/bin/claude')
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHECKS = ['basic', 'max-tokens', 'search', 'thinking', 'pdf']
SERVER_TOOL = re.compile(r'^web_search_\d{8}$')
HOP_HEADERS = {'host', 'content-length', 'accept-encoding', 'connection', 'transfer-encoding', 'keep-alive'}


def document_blocks(value):
    """How many Anthropic document blocks a message list holds, inside tool results too."""
    if isinstance(value, list):
        return sum(document_blocks(item) for item in value)
    if isinstance(value, dict):
        return (value.get('type') == 'document') + document_blocks(value.get('content'))
    return 0


def request_fields(path, headers, raw):
    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = {}
    auth = headers.get('authorization', '')
    return {'path': path, 'model': body.get('model'), 'max_tokens': body.get('max_tokens'),
            'stream': body.get('stream'), 'thinking': body.get('thinking'),
            'output_config': body.get('output_config'),
            'tools': [tool.get('type') or 'custom' for tool in body.get('tools') or [] if isinstance(tool, dict)],
            'documents': document_blocks(body.get('messages')),
            'auth': 'bearer' if auth.startswith('Bearer ') else ('x-api-key' if 'x-api-key' in headers else None)}


def response_fields(status, content_type, raw):
    fields = {'status': status}
    text = []
    if 'text/event-stream' in content_type:
        for line in raw.decode(errors='replace').splitlines():
            if not line.startswith('data:'):
                continue
            try:
                event = json.loads(line[5:])
            except ValueError:
                continue
            if event.get('type') == 'message_delta':
                fields.update(stop_reason=event['delta'].get('stop_reason'), usage=event.get('usage'))
            elif event.get('type') == 'content_block_delta' and event['delta'].get('type') == 'text_delta':
                text.append(event['delta']['text'])
            elif event.get('type') == 'error':
                fields['error'] = str(event.get('error', {}).get('message', ''))[:300]
    else:
        try:
            body = json.loads(raw)
        except ValueError:
            body = {}
        if status >= 400:
            fields['error'] = json.dumps(body.get('error', body))[:300]
        fields.update(stop_reason=body.get('stop_reason'), usage=body.get('usage'))
        text = [block.get('text', '') for block in body.get('content') or [] if isinstance(block, dict)]
    fields['reply_has_url'] = 'https://' in ''.join(text)
    return fields


class Relay(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, upstream):
        self.upstream = upstream
        self.records = []
        self.lock = threading.Lock()
        super().__init__(('127.0.0.1', 0), RelayHandler)


class RelayHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def forward(self):
        length = int(self.headers.get('Content-Length') or 0)
        raw = self.rfile.read(length) if length else b''
        headers = {name.lower(): value for name, value in self.headers.items()}
        record = {'request': request_fields(self.path, headers, raw)}
        req = urllib.request.Request(self.server.upstream + self.path, data=raw or None, method=self.command,
                                     headers={k: v for k, v in headers.items() if k not in HOP_HEADERS})
        try:
            response = OPENER.open(req, timeout=1800)
        except urllib.error.HTTPError as error:
            response = error
        body = []
        with response:
            self.send_response(response.status)
            for name, value in response.headers.items():
                if name.lower() not in HOP_HEADERS:
                    self.send_header(name, value)
            self.send_header('Connection', 'close')
            self.end_headers()
            while chunk := response.read1(65536):
                body.append(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
        self.close_connection = True
        record['response'] = response_fields(response.status, response.headers.get('Content-Type', ''), b''.join(body))
        with self.server.lock:
            self.server.records.append(record)

    do_GET = do_POST = do_HEAD = forward


def run_claude(workdir, relay, model, key, prompt, tools, extra_env=None, timeout=900):
    home = workdir / 'home'
    (home / 'config').mkdir(parents=True, exist_ok=True)
    (workdir / 'cwd').mkdir(exist_ok=True)
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(home), 'CLAUDE_CONFIG_DIR': str(home / 'config'),
           'TMPDIR': str(home), 'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{relay.server_address[1]}',
           'ANTHROPIC_AUTH_TOKEN': key, 'ANTHROPIC_MODEL': model, 'ANTHROPIC_SMALL_FAST_MODEL': model,
           'ANTHROPIC_DEFAULT_HAIKU_MODEL': model, 'ANTHROPIC_DEFAULT_SONNET_MODEL': model,
           'ANTHROPIC_DEFAULT_OPUS_MODEL': model, 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
           'DISABLE_AUTOUPDATER': '1', 'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1', **(extra_env or {})}
    command = [str(CLAUDE), '-p', prompt, '--model', model, '--output-format', 'json', '--no-session-persistence',
               '--setting-sources', 'user', '--tools', ','.join(tools)]
    if tools:
        command += ['--allowedTools', ','.join(tools)]
    start = len(relay.records)
    proc = subprocess.run(command, cwd=workdir / 'cwd', env=env, capture_output=True, text=True, timeout=timeout)
    try:
        output = json.loads(proc.stdout)
    except ValueError:
        output = {'raw_stdout': proc.stdout[-500:]}
    result = {'exit_code': proc.returncode, 'is_error': output.get('is_error'), 'subtype': output.get('subtype'),
              'result': str(output.get('result', output.get('raw_stdout', '')))[:500],
              'stderr_tail': proc.stderr[-300:], 'requests': relay.records[start:]}
    return result


def messages_calls(result):
    return [r for r in result['requests'] if r['request']['path'].split('?')[0] == '/v1/messages']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='+', choices=CHECKS, default=CHECKS)
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'serving.json').read_text())
    model, key = cfg['served_model_name'], (ROOT / '.secrets/api-key').read_text().strip()
    relay = Relay(f'http://{cfg["bind_host"]}:{cfg["port"]}')
    threading.Thread(target=relay.serve_forever, daemon=True).start()
    out = ROOT / '.runs/claude-code' / f'{datetime.datetime.now():%Y%m%d-%H%M%S}'
    version = subprocess.run([str(CLAUDE), '--version'], capture_output=True, text=True).stdout.strip()
    results = {'claude_version': version}
    for name in args.only:
        workdir = out / name
        if name == 'basic':
            run = run_claude(workdir, relay, model, key, 'Reply with exactly: KIMI_CC_OK', [])
            calls = messages_calls(run)
            ok = run['exit_code'] == 0 and 'KIMI_CC_OK' in run['result'] and all(
                c['response']['status'] == 200 for c in calls)
        elif name == 'max-tokens':
            run = run_claude(workdir, relay, model, key, 'Write a 3000-word essay on the history of bridges.', [],
                             {'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '256'})
            calls = [c for c in messages_calls(run) if c['response']['status'] == 200]
            limits = [c['request']['max_tokens'] for c in calls]
            used = [(c['response'].get('usage') or {}).get('output_tokens') for c in calls]
            stopped = [c['response'].get('stop_reason') for c in calls]
            ok = bool(calls) and all(limit == 256 for limit in limits) and all(
                u is not None and u <= 256 for u in used) and 'max_tokens' in stopped
            run['summary'] = {'max_tokens_sent': limits, 'output_tokens': used, 'stop_reasons': stopped}
        elif name == 'search':
            run = run_claude(workdir, relay, model, key,
                             'Use the WebSearch tool to find the GitHub repository of the SGLang inference engine, '
                             'then reply with its URL only.', ['WebSearch'])
            searches = [c for c in messages_calls(run) if any(SERVER_TOOL.fullmatch(t) for t in c['request']['tools'])]
            ok = (run['exit_code'] == 0 and bool(searches) and all(
                c['response']['status'] == 200 and c['response']['reply_has_url'] for c in searches)
                and 'github.com' in run['result'].lower())
            run['summary'] = {'search_subrequests': len(searches),
                              'tool_types': sorted({t for c in searches for t in c['request']['tools']})}
        elif name == 'pdf':
            from test_documents import fixture_pdf
            code, visual = 'TEXT_' + secrets.token_hex(3), 'VISUAL_' + secrets.token_hex(3)
            (workdir / 'cwd').mkdir(parents=True, exist_ok=True)
            (workdir / 'cwd/report.pdf').write_bytes(fixture_pdf(code, visual))
            run = run_claude(workdir, relay, model, key,
                             'Use the Read tool to read report.pdf, both pages. Reply with the code on page 1, '
                             'the code on page 2, and which bar in the chart is taller.', ['Read'])
            calls = messages_calls(run)
            reads = [c for c in calls if c['request']['documents']]
            ok = (run['exit_code'] == 0 and bool(reads) and all(c['response']['status'] == 200 for c in calls)
                  and code in run['result'] and visual in run['result'] and 'red' in run['result'].lower())
            run['summary'] = {'requests_with_documents': len(reads), 'statuses': [c['response']['status'] for c in calls],
                              'result': run['result'][:200]}
        else:
            run = run_claude(workdir, relay, model, key, 'How many positive divisors does 360 have? Reply with the number.',
                             [], {'MAX_THINKING_TOKENS': '2048'})
            calls = messages_calls(run)
            ok = run['exit_code'] == 0 and '24' in run['result']
            run['summary'] = {'thinking_sent': [c['request']['thinking'] for c in calls],
                              'output_config_sent': [c['request']['output_config'] for c in calls]}
        results[name] = dict(run, ok=ok)
        shown = run.get('summary') or {'result': run['result'][:120]}
        print(f'{"ok  " if ok else "FAIL"} {name}: exit {run["exit_code"]}, {len(messages_calls(run))} /v1/messages '
              f'requests; {json.dumps(shown, ensure_ascii=False)[:900]}', flush=True)
    relay.shutdown()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    failed = [name for name in args.only if not results[name]['ok']]
    print(f'\n{len(failed)} failed: {", ".join(failed)}' if failed else '\nall checks passed', f'({out / "summary.json"})')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
