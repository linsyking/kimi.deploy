"""Per-request replacement of Kimi's internal thinking-effort prompt.

When thinking is on, the K3 encoder renders a `thinking-effort` system message before the
user's system prompt. A request replaces that whole message with
`chat_template_kwargs.thinking_effort_prompt` (OpenAI), a top-level `thinking_effort_prompt`
(Anthropic), or the percent-encoded `X-Thinking-Effort-Prompt` header (either route; the body
wins). `{effort}` becomes the tier the encoder renders: low, high or max. kimictl loads this in
every engine process; requests without an override render the stock message.
"""
from contextvars import ContextVar
from functools import wraps
import json
import sys
from urllib.parse import unquote

FIELD = 'thinking_effort_prompt'
HEADER = 'x-thinking-effort-prompt'
# Set only around one synchronous encoder call, which may run in an executor thread.
_rendered = ContextVar('kimi_rendered_effort_prompt', default=None)
# Set by the Anthropic entry points around their synchronous request conversion.
_anthropic = ContextVar('kimi_anthropic_effort_prompt', default=None)


def validated(prompt):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f'{FIELD} must be a non-empty string')
    return prompt


def header_prompt(raw_request):
    value = raw_request.headers.get(HEADER) if raw_request is not None else None
    return None if value is None else unquote(value)


async def anthropic_prompt(raw_request):
    body = await raw_request.body()
    if FIELD.encode() in body:
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get(FIELD) is not None:
            return parsed[FIELD]
    return header_prompt(raw_request)


def with_prompt(template_kwargs, prompt):
    return {**(template_kwargs or {}), FIELD: prompt}


def patch_tokenizer(tokenizer_module):
    build = tokenizer_module.build_chat_segments
    if getattr(build, '_kimi_effort_prompt', False):
        return
    encoding = sys.modules[build.__module__]
    message = encoding._internal_system_message

    @wraps(message)
    def internal_system_message(message_type, body):
        rendered = _rendered.get()
        if message_type == 'thinking-effort' and rendered is not None:
            body = rendered
        return message(message_type, body)

    @wraps(build)
    def build_chat_segments(*args, **kwargs):
        prompt = kwargs.pop(FIELD, None)
        if prompt is None:
            return build(*args, **kwargs)
        effort = kwargs.get('thinking_effort')
        token = _rendered.set(validated(prompt).replace('{effort}', str(effort)))
        try:
            return build(*args, **kwargs)
        finally:
            _rendered.reset(token)

    build_chat_segments._kimi_effort_prompt = True
    encoding._internal_system_message = internal_system_message
    tokenizer_module.build_chat_segments = build_chat_segments


def install():
    from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.entrypoints.openai.serving_kimi_k3 import KimiK3ChatEncodingAdapter

    init = KimiK3ChatEncodingAdapter.__init__
    if getattr(init, '_kimi_effort_prompt', False):
        return

    @wraps(init)
    def adapter_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        patch_tokenizer(sys.modules[type(self._tokenizer).__module__])

    handle_request = OpenAIServingChat.handle_request

    @wraps(handle_request)
    async def openai_handle_request(self, request, raw_request):
        prompt = header_prompt(raw_request)
        if prompt is not None and FIELD not in (request.chat_template_kwargs or {}):
            request.chat_template_kwargs = with_prompt(request.chat_template_kwargs, prompt)
        return await handle_request(self, request, raw_request)

    convert = AnthropicServing._convert_to_chat_completion_request

    @wraps(convert)
    def anthropic_convert(self, request):
        chat = convert(self, request)
        prompt = _anthropic.get()
        if prompt is not None:
            chat.chat_template_kwargs = with_prompt(chat.chat_template_kwargs, prompt)
        return chat

    def anthropic_entry(handler):
        @wraps(handler)
        async def entry(self, request, raw_request):
            token = _anthropic.set(await anthropic_prompt(raw_request))
            try:
                return await handler(self, request, raw_request)
            finally:
                _anthropic.reset(token)
        return entry

    adapter_init._kimi_effort_prompt = True
    KimiK3ChatEncodingAdapter.__init__ = adapter_init
    OpenAIServingChat.handle_request = openai_handle_request
    AnthropicServing._convert_to_chat_completion_request = anthropic_convert
    AnthropicServing.handle_messages = anthropic_entry(AnthropicServing.handle_messages)
    AnthropicServing.handle_count_tokens = anthropic_entry(AnthropicServing.handle_count_tokens)
