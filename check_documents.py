#!/usr/bin/env python3
"""Live PDF regression against this deployment (JSON, SSE, nested Read results), from ds.deploy.

Beyond ds.deploy's cases it checks that citations get a 400, that web search with a PDF attached
searches the user's text and not page text, and measures what a PDF page costs in input tokens
(1- and 10-page PDFs of text pages and of blank pages). Run on c04 with the serving venv's Python,
which has Pillow for the fixture. Evidence goes to audit/documents-<time>/.

    .venvs/sglang-7422c70f/bin/python check_documents.py [--skip-cost]"""
import argparse
import datetime
import json
from pathlib import Path
import runpy
import secrets
import time
import urllib.error
import urllib.request

from anthropic_check import Client, text
from test_documents import document, fixture_pdf, payload

ROOT = Path(__file__).resolve().parent
QUESTION = ('Read both pages of the PDF. Return the code on page 1, the code on page 2, '
            'and which bar is taller in the chart. Answer briefly.')
WORDS = ('harbour ferry timetable crossing tide cargo berth pilot schedule weather lock river '
         'archive ledger station market orchard printing glassworks tram signal').split()


def read_pdf(client, pdf, code, visual, nested, stream):
    """One PDF read through /v1/messages, as an attachment or inside a Read tool result."""
    body = payload(document(pdf, title='PDF vision regression'), nested=nested)
    for key in ('model', 'stream'):  # Client adds both
        body.pop(key)
    body.update(max_tokens=512, temperature=0, thinking={'type': 'disabled'})
    if nested:
        # Anthropic requires tool results first in the responding user
        # turn; text before a pending result violates tool-call order.
        body['messages'][-1]['content'].pop(0)
        body['messages'][0]['content'] = QUESTION
        body['messages'][-1]['content'][-1]['text'] = QUESTION
    else:
        body['messages'][0]['content'][0]['text'] = QUESTION
    response = client.stream(**body) if stream else client.messages(**body)
    answer = text(response)
    ended = response['stopped'] if stream else response.get('stop_reason') == 'end_turn'
    return code in answer and visual in answer and 'red' in answer.lower() and ended, response


def error_response(cfg, ctl, body):
    """Status and JSON of a request expected to fail before inference."""
    req = urllib.request.Request(f'http://{cfg["bind_host"]}:{cfg["port"]}/v1/messages',
                                 data=json.dumps(body).encode(),
                                 headers=dict(ctl['auth_headers'](), **{'Content-Type': 'application/json'}))
    try:
        ctl['OPENER'].open(req, timeout=60).close()
        return 200, {}
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def pages_pdf(count, lines):
    """A letter-size PDF of `count` pages, each with `lines` lines of Helvetica text (0: blank)."""
    objects = ['<< /Type /Catalog /Pages 2 0 R >>',
               f'<< /Type /Pages /Count {count} /Kids [{" ".join(f"{4 + 2 * i} 0 R" for i in range(count))}] >>',
               '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    for page in range(count):
        rows = [' '.join(WORDS[(page * 7 + row * 3 + k) % len(WORDS)] for k in range(12)) for row in range(lines)]
        content = 'BT /F1 11 Tf 14 TL 50 742 Td ' + ' '.join(f'({row}) Tj T*' for row in rows) + ' ET'
        objects.append(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] '
                       f'/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * page} 0 R >>')
        objects.append(f'<< /Length {len(content)} >>\nstream\n{content}\nendstream')
    result, offsets = bytearray(b'%PDF-1.4\n'), []
    for number, value in enumerate(objects, 1):
        offsets.append(len(result))
        result.extend(f'{number} 0 obj\n{value}\nendobj\n'.encode())
    start = len(result)
    result.extend(f'xref\n0 {len(objects) + 1}\n0000000000 65535 f \n'.encode())
    for offset in offsets:
        result.extend(f'{offset:010d} 00000 n \n'.encode())
    result.extend(f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF'.encode())
    return bytes(result)


def page_cost(client, context_length):
    """input_tokens for 1- and 10-page PDFs; the difference over 9 pages is the cost of one page."""
    counts = {}
    for kind, lines in (('text', 48), ('blank', 0)):
        for count in (1, 10):
            response = client.messages(max_tokens=16, temperature=0, thinking={'type': 'disabled'}, messages=[
                {'role': 'user', 'content': [{'type': 'text', 'text': 'Reply with OK.'},
                                             document(pages_pdf(count, lines))]}])
            counts[f'{kind}_{count}'] = response['usage']['input_tokens']
    result = {'input_tokens': counts}
    for kind in ('text', 'blank'):
        per_page = (counts[f'{kind}_10'] - counts[f'{kind}_1']) / 9
        result[f'{kind}_tokens_per_page'] = round(per_page, 1)
        result[f'{kind}_pages_in_context'] = int((context_length - counts[f'{kind}_1'] + per_page) // per_page)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--skip-cost', action='store_true', help='skip the page-cost measurement')
    args = parser.parse_args()
    ctl = runpy.run_path(str(ROOT / 'kimictl'), run_name='document_smoke')
    cfg = ctl['config']()
    client = Client()
    destination = ROOT / 'audit' / ('documents-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    destination.mkdir(parents=True, exist_ok=False)
    code, visual = 'TEXT_' + secrets.token_hex(3), 'VISUAL_' + secrets.token_hex(3)
    pdf = fixture_pdf(code, visual)
    (destination / 'fixture.pdf').write_bytes(pdf)
    results = []
    def record(result):
        results.append(result)
        print(f'{result["name"]}: {"PASS" if result["passed"] else "FAIL"}', flush=True)
    for nested in (False, True):
        for stream in (False, True):
            name = f'{"tool_result" if nested else "attachment"}_{"stream" if stream else "json"}'
            started = time.monotonic()
            try:
                passed, response = read_pdf(client, pdf, code, visual, nested, stream)
                record(dict(name=name, passed=passed, elapsed_seconds=round(time.monotonic() - started, 3),
                            response=response))
            except Exception as error:
                record(dict(name=name, passed=False, error=str(error)))
    # Counting must accept exactly the same document shape, not just generation.
    body = payload(document(pdf), nested=True)
    body['messages'][-1]['content'].pop(0)
    for key in ('stream', 'max_tokens'):
        body.pop(key)
    body['model'] = cfg['served_model_name']
    try:
        with client.post('/v1/messages/count_tokens', body) as response:
            counted = json.load(response)
        record(dict(name='count_tokens', passed=counted.get('input_tokens', 0) > 0, response=counted))
    except Exception as error:
        record(dict(name='count_tokens', passed=False, error=str(error)))
    # Error responses must be explicit 400s rather than Pydantic union errors.
    for name, block in (('invalid_pdf', document(b'not a PDF')), ('citations', document(pdf, citations={'enabled': True}))):
        invalid = payload(block)
        invalid['model'] = cfg['served_model_name']
        status, response = error_response(cfg, ctl, invalid)
        record(dict(name=name, passed=status == 400 and response.get('error', {}).get('type') == 'invalid_request_error',
                    status=status, response=response))
    # Search must take its query from the user's text even with page text in the same turn. Exa
    # queries are not logged, so judge by the sources: the user's query finds SGLang's docs or
    # repository; a query made of page text ("Page 2/2 ... read the page image") would not.
    query = 'SGLang Anthropic document support'
    try:
        response = client.messages(
            max_tokens=1024, temperature=0, thinking={'type': 'disabled'},
            system='Answer in one sentence and cite one source URL from the search evidence.',
            messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'Perform a web search for the query: ' + query},
                                                   document(pdf)]}],
            tools=[{'type': 'web_search_20250305', 'name': 'web_search'}], tool_choice={'type': 'tool', 'name': 'web_search'})
        answer = text(response)
        on_topic = any(source in answer for source in ('docs.sglang', 'github.com/sgl-project'))
        record(dict(name='search_query', passed=on_topic and code not in answer, answer=answer))
    except Exception as error:
        record(dict(name='search_query', passed=False, error=str(error)))
    cost = None
    if not args.skip_cost:
        cost = page_cost(client, cfg['context_length'])
        print('page cost: ' + json.dumps(cost), flush=True)
    (destination / 'summary.json').write_text(json.dumps(dict(expected_text=code, expected_visual=visual,
        results=results, page_cost=cost), indent=2, ensure_ascii=False) + '\n')
    print('Evidence: ' + str(destination), flush=True)
    return 0 if all(r['passed'] for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
