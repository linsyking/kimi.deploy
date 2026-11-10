#!/usr/bin/env python3
"""Functional checks against the running endpoint (the brief's validation checklist, item 3):
/health and /v1/models, chat at reasoning_effort low and max, a tool call round trip, an image
input, and a long-context needle. Stdlib only; run on c04. Results go to .runs/smoke/.

Requests carry no sampling overrides: the model fixes temperature 1.0 and top_p 0.95.

    smoke_test.py [--only health chat-low ...] [--needle-tokens 128000]"""
import argparse
import base64
import datetime
import json
from pathlib import Path
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parent
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
CHECKS = ['health', 'chat-low', 'chat-max', 'tool', 'image', 'needle']


class Client:
    def __init__(self):
        cfg = json.loads((ROOT / 'serving.json').read_text())
        self.base = f'http://{cfg["bind_host"]}:{cfg["port"]}'
        self.model = cfg['served_model_name']
        self.key = (ROOT / '.secrets/api-key').read_text().strip()

    def call(self, path, body=None, timeout=60):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, headers={
            'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
        try:
            with OPENER.open(req, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f'{path}: HTTP {exc.code}: {exc.read().decode(errors="replace")[:500]}')
        return json.loads(raw) if raw.strip().startswith((b'{', b'[')) else raw.decode()

    def chat(self, messages, timeout=900, **extra):
        started = time.monotonic()
        response = self.call('/v1/chat/completions', {'model': self.model, 'messages': messages, **extra},
                             timeout=timeout)
        response['elapsed_s'] = round(time.monotonic() - started, 2)
        return response


def message(response):
    choice = response['choices'][0]
    msg = choice['message']
    return msg, choice['finish_reason'], msg.get('reasoning_content') or msg.get('reasoning') or ''


def summary(response, **fields):
    msg, finish, reasoning = message(response)
    usage = response.get('usage') or {}
    return {'finish_reason': finish, 'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'), 'reasoning_chars': len(reasoning),
            'elapsed_s': response['elapsed_s'], 'content': (msg.get('content') or '')[:300], **fields}


def check_health(client):
    client.call('/health', timeout=10)
    models = client.call('/v1/models', timeout=10)
    ids = [item['id'] for item in models['data']]
    return client.model in ids, {'models': ids, 'max_model_len': models['data'][0].get('max_model_len')}


def check_effort(client, effort):
    response = client.chat([{'role': 'user', 'content':
                             'How many positive divisors does 360 have? Reply with just the number.'}],
                           reasoning_effort=effort, max_tokens=32768)
    msg, finish, reasoning = message(response)
    return finish == 'stop' and (msg.get('content') or '').strip().rstrip('.') == '24' and bool(reasoning), \
        summary(response, reasoning_effort=effort)


WEATHER_TOOL = {'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Current weather for a city.',
    'parameters': {'type': 'object', 'required': ['city'], 'properties': {
        'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['celsius', 'fahrenheit']}}}}}


def check_tool(client):
    messages = [{'role': 'user', 'content': 'What is the weather in Paris right now, in celsius?'}]
    first = client.chat(messages, tools=[WEATHER_TOOL], reasoning_effort='low', max_tokens=8192)
    msg, finish, _ = message(first)
    calls = msg.get('tool_calls') or []
    if finish != 'tool_calls' or len(calls) != 1 or calls[0]['function']['name'] != 'get_weather':
        return False, summary(first, tool_calls=calls)
    arguments = json.loads(calls[0]['function']['arguments'])
    messages += [msg, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                       'content': json.dumps({'city': 'Paris', 'temperature_c': 17, 'condition': 'light rain'})}]
    second = client.chat(messages, tools=[WEATHER_TOOL], reasoning_effort='low', max_tokens=8192)
    answer, finish, _ = message(second)
    ok = 'paris' in arguments.get('city', '').lower() and finish == 'stop' and '17' in (answer.get('content') or '')
    return ok, summary(second, arguments=arguments)


def png(width, height, pixel):
    rows = b''.join(b'\0' + b''.join(bytes(pixel(x, y)) for x in range(width)) for y in range(height))

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def check_image(client):
    image = png(256, 256, lambda x, y: (220, 20, 20) if x < 128 else (20, 40, 220))
    url = 'data:image/png;base64,' + base64.b64encode(image).decode()
    response = client.chat([{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': url}},
        {'type': 'text', 'text': 'The image has a left half and a right half of different colors. '
                                 'Answer exactly in the form "left: <color>, right: <color>".'}]}],
        reasoning_effort='low', max_tokens=8192)
    content = (message(response)[0].get('content') or '').lower()
    left, _, right = content.partition('right')
    return 'red' in left and 'blue' in right, summary(response)


NAMES = ['Arden', 'Brisk', 'Calloway', 'Dunmore', 'Elsby', 'Farrow', 'Galt', 'Hensley', 'Ivers', 'Jessup']
SUBJECTS = ['the harbour office', 'a tram line', 'the grain market', 'a weather station', 'the town archive',
            'a ferry crossing', 'the glassworks', 'a hillside orchard', 'the river lock', 'a printing house']
VERBS = ['recorded', 'repaired', 'expanded', 'closed', 'inspected', 'rebuilt', 'catalogued', 'moved']


def haystack(rng, sentences):
    return ' '.join(f'In {rng.randint(1780, 1990)}, the people of {rng.choice(NAMES)} {rng.choice(VERBS)} '
                    f'{rng.choice(SUBJECTS)} and counted {rng.randint(2, 9000)} visitors that season.'
                    for _ in range(sentences))


def check_needle(client, target_tokens):
    question = 'What is the access code for the Lindqvist vault? Reply with the code only.'
    # Calibrate tokens per sentence on a short sample, then size the haystack to the target.
    sample = client.chat([{'role': 'user', 'content': haystack(random.Random(1), 2000)}], max_tokens=1)
    per_sentence = sample['usage']['prompt_tokens'] / 2000
    rng = random.Random(0)
    count = int(target_tokens / per_sentence)
    needle_at = int(count * 0.6)
    text = (haystack(rng, needle_at) + ' The access code for the Lindqvist vault is 4817-QUARTZ. '
            + haystack(rng, count - needle_at))
    response = client.chat([{'role': 'user', 'content': text + '\n\n' + question}],
                           reasoning_effort='low', max_tokens=8192, timeout=1800)
    msg, finish, _ = message(response)
    return '4817-QUARTZ' in (msg.get('content') or '').upper(), summary(response, needle_depth=0.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--only', nargs='+', choices=CHECKS, default=CHECKS)
    parser.add_argument('--needle-tokens', type=int, default=128000)
    args = parser.parse_args()
    client = Client()
    runs = {'health': lambda: check_health(client), 'chat-low': lambda: check_effort(client, 'low'),
            'chat-max': lambda: check_effort(client, 'max'), 'tool': lambda: check_tool(client),
            'image': lambda: check_image(client), 'needle': lambda: check_needle(client, args.needle_tokens)}
    results = {}
    for name in args.only:
        try:
            ok, detail = runs[name]()
        except Exception as exc:
            ok, detail = False, {'error': f'{type(exc).__name__}: {exc}'}
        results[name] = {'ok': ok, **detail}
        print(f'{"ok  " if ok else "FAIL"} {name}: {json.dumps(detail, ensure_ascii=False)}', flush=True)
    out = ROOT / '.runs/smoke' / f'{datetime.datetime.now():%Y%m%d-%H%M%S}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    failed = [name for name, result in results.items() if not result['ok']]
    print(f'\n{len(failed)} failed: {", ".join(failed)}' if failed else '\nall checks passed', f'({out})')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
