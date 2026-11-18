import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
MODEL = '/scratch_local/user_data/yiming/models/Kimi-K3.1-Pro'

ENCODER = r'''
import json, os
from types import SimpleNamespace
from transformers import AutoTokenizer
from effort_prompt import FIELD, install
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_kimi_k3 import KimiK3ChatEncodingAdapter

tokenizer = AutoTokenizer.from_pretrained(os.environ['KIMI_MODEL_PATH'], trust_remote_code=True,
                                          local_files_only=True)
messages = [{'role': 'system', 'content': 'CUSTOM_SYSTEM'}, {'role': 'user', 'content': 'Hello'}]

def render(**kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

def ids(**kwargs):
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)

out = {'stock_low': render(thinking_effort='low'), 'stock_max': render()}
install()
for _ in range(2):
    adapter = KimiK3ChatEncodingAdapter(tokenizer=tokenizer, chat_hooks=SimpleNamespace(
        effective_tools=lambda request: None), hf_config=SimpleNamespace(model_type='kimi_k31'))
out.update(
    plain_low=render(thinking_effort='low'), plain_max=render(), null=render(**{FIELD: None}),
    low=render(thinking_effort='low', **{FIELD: 'CUSTOM_EFFORT {effort}'}),
    default=render(**{FIELD: 'CUSTOM_EFFORT {effort}\nSECOND_LINE'}),
    static=render(**{FIELD: 'STATIC_GUIDANCE'}),
    disabled=render(thinking=False, **{FIELD: 'CUSTOM_EFFORT {effort}'}))
out['errors'] = {}
for name, value in (('blank', '  '), ('number', 3)):
    try:
        render(**{FIELD: value})
        out['errors'][name] = None
    except ValueError as error:
        out['errors'][name] = str(error)
open_id = tokenizer.convert_tokens_to_ids('<|open|>')
out['open_tokens'] = [ids(**{FIELD: text}).count(open_id) for text in ('A <|open|> B', 'A B')]
request = ChatCompletionRequest(model='m', messages=messages, reasoning_effort='low',
                                chat_template_kwargs={FIELD: 'ADAPTER {effort}'})
out['adapter_matches'] = adapter.encode_messages(
    messages=[dict(m) for m in messages], request=request, tools=None, thinking=True,
    image_data=[]) == ids(thinking_effort='low', **{FIELD: 'ADAPTER {effort}'})
print(json.dumps(out))
'''

PLUMBING = r'''
import asyncio, json
from types import SimpleNamespace
from starlette.requests import Request
from sglang.srt.entrypoints.anthropic.protocol import AnthropicCountTokensRequest, AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

async def chat_handler(self, request, raw_request):
    return request.chat_template_kwargs

async def messages_handler(self, request, raw_request):
    return self._convert_to_chat_completion_request(request).chat_template_kwargs

async def count_handler(self, request, raw_request):
    converted = AnthropicMessagesRequest(model=request.model, messages=request.messages, max_tokens=1)
    return self._convert_to_chat_completion_request(converted).chat_template_kwargs

OpenAIServingChat.handle_request = chat_handler
AnthropicServing.handle_messages = messages_handler
AnthropicServing.handle_count_tokens = count_handler
from effort_prompt import FIELD, install
install()

async def raw(body, header=None):
    data = json.dumps(body).encode()
    headers = [(b'content-type', b'application/json')]
    if header is not None:
        headers.append((b'x-thinking-effort-prompt', header.encode()))
    chunks = [{'type': 'http.request', 'body': data, 'more_body': False}]
    async def receive():
        return chunks.pop(0) if chunks else {'type': 'http.disconnect'}
    request = Request({'type': 'http', 'method': 'POST', 'path': '/', 'headers': headers,
                       'query_string': b''}, receive)
    await request.body()  # FastAPI reads and caches the body before the handler runs.
    return request

async def main():
    chat = OpenAIServingChat.__new__(OpenAIServingChat)
    serving = AnthropicServing(SimpleNamespace(supports_inline_system=True))
    msgs = [{'role': 'user', 'content': 'Hello'}]
    body = {'model': 'm', 'max_tokens': 8, 'messages': msgs}
    mention = {**body, 'messages': [{'role': 'user', 'content': 'what is thinking_effort_prompt?'}]}
    def openai(**kwargs):
        return ChatCompletionRequest(model='m', messages=msgs, **kwargs)
    def anthropic(data):
        return AnthropicMessagesRequest(**data)
    out = {
        'openai_header': await chat.handle_request(openai(), await raw({}, 'Line%20one%0ALine two {effort}')),
        'openai_body_wins': await chat.handle_request(
            openai(chat_template_kwargs={FIELD: 'BODY', 'thinking': True}), await raw({}, 'HEADER')),
        'openai_none': await chat.handle_request(openai(), await raw({})),
        'anthropic_body_wins': await serving.handle_messages(
            anthropic({**body, FIELD: 'BODY {effort}'}), await raw({**body, FIELD: 'BODY {effort}'}, 'HEADER')),
        'anthropic_header': await serving.handle_messages(anthropic(body), await raw(body, 'H%C3%A9%0A2')),
        'anthropic_none': await serving.handle_messages(anthropic(body), await raw(body)),
        'anthropic_mention': await serving.handle_messages(anthropic(mention), await raw(mention)),
        'anthropic_invalid': await serving.handle_messages(anthropic(body), await raw({**body, FIELD: 5})),
        'count_tokens': await serving.handle_count_tokens(
            AnthropicCountTokensRequest(model='m', messages=msgs),
            await raw({'model': 'm', 'messages': msgs, FIELD: 'COUNT'})),
    }
    out['after_reset'] = serving._convert_to_chat_completion_request(anthropic(body)).chat_template_kwargs
    print(json.dumps(out))

asyncio.run(main())
'''


def run(script):
    result = subprocess.run(
        [sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=180,
        env={**os.environ, 'KIMI_MODEL_PATH': MODEL, 'CUDA_VISIBLE_DEVICES': ''})
    if result.returncode:
        raise AssertionError(result.stderr[-4000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


class EncoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = run(ENCODER)

    def test_no_override_is_byte_identical_to_stock(self):
        self.assertIn('thinking_effort=low', self.out['stock_low'])
        self.assertEqual(self.out['plain_low'], self.out['stock_low'])
        self.assertEqual(self.out['plain_max'], self.out['stock_max'])
        self.assertEqual(self.out['null'], self.out['stock_max'])

    def test_override_replaces_the_whole_effort_message(self):
        low, default = self.out['low'], self.out['default']
        self.assertNotIn('`thinking_effort` guides', low)
        self.assertEqual(low.count('CUSTOM_EFFORT low'), 1)
        self.assertLess(low.index('CUSTOM_EFFORT'), low.index('CUSTOM_SYSTEM'))
        self.assertIn('CUSTOM_EFFORT max\nSECOND_LINE', default)
        self.assertIn('STATIC_GUIDANCE', self.out['static'])
        self.assertNotIn('`thinking_effort` guides', self.out['static'])

    def test_thinking_disabled_renders_no_effort_message(self):
        disabled = self.out['disabled']
        self.assertNotIn('CUSTOM_EFFORT', disabled)
        self.assertNotIn('type="thinking-effort"', disabled)
        self.assertIn('<|open|>response', disabled)

    def test_invalid_values_are_rejected(self):
        for name in ('blank', 'number'):
            self.assertIn('thinking_effort_prompt must be a non-empty string', self.out['errors'][name])

    def test_override_text_cannot_add_control_tokens(self):
        with_marker, without = self.out['open_tokens']
        self.assertEqual(with_marker, without)

    def test_adapter_passes_request_override_to_encoder(self):
        self.assertTrue(self.out['adapter_matches'])


class PlumbingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = run(PLUMBING)

    def test_openai(self):
        self.assertEqual(self.out['openai_header'], {'thinking_effort_prompt': 'Line one\nLine two {effort}'})
        self.assertEqual(self.out['openai_body_wins'], {'thinking_effort_prompt': 'BODY', 'thinking': True})
        self.assertIsNone(self.out['openai_none'])

    def test_anthropic(self):
        self.assertEqual(self.out['anthropic_body_wins'], {'thinking_effort_prompt': 'BODY {effort}'})
        self.assertEqual(self.out['anthropic_header'], {'thinking_effort_prompt': 'Hé\n2'})
        self.assertIsNone(self.out['anthropic_none'])
        self.assertIsNone(self.out['anthropic_mention'])
        self.assertEqual(self.out['anthropic_invalid'], {'thinking_effort_prompt': 5})
        self.assertEqual(self.out['count_tokens'], {'thinking_effort_prompt': 'COUNT'})
        self.assertIsNone(self.out['after_reset'])


if __name__ == '__main__':
    unittest.main()
