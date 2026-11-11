"""Offline checks of thinking_budget.py against the installed SGLang (no GPU, no server).

The patched methods' originals are replaced by recorders before install(), so these tests
exercise only the adapter: the budget's path into custom_params, the request normalization,
and which requests reach the grammar path.

    CUDA_VISIBLE_DEVICES= .venvs/sglang/bin/python -m unittest -v test_thinking_budget.py"""
from types import SimpleNamespace
import unittest

from sglang.srt.constrained.grammar_manager import GrammarManager
from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing

CONVERTED, PROCESSED = [], []


def fake_convert(self, request):
    CONVERTED.append(request)
    return SimpleNamespace(custom_params={'kept': 1})


def fake_process(self, req):
    PROCESSED.append(req)
    return True


AnthropicServing._convert_to_chat_completion_request = fake_convert
GrammarManager.process_req_with_grammar = fake_process
import thinking_budget  # noqa: E402
thinking_budget.install()
thinking_budget.install()  # idempotent


def request(**thinking):
    body = {'model': 'kimi-k3.1-pro', 'max_tokens': 4096, 'messages': [{'role': 'user', 'content': 'hi'}],
            'output_config': {'effort': 'high'}}
    if thinking:
        body['thinking'] = thinking
    return AnthropicMessagesRequest(**body)


def req(**params):
    fields = dict(json_schema=None, regex=None, ebnf=None, structural_tag=None, custom_params=None)
    return SimpleNamespace(sampling_params=SimpleNamespace(**dict(fields, **params)))


class ConvertTests(unittest.TestCase):
    def setUp(self):
        CONVERTED.clear()

    def test_budget_becomes_custom_param(self):
        chat = AnthropicServing._convert_to_chat_completion_request(None, request(type='enabled', budget_tokens=2048))
        self.assertEqual(chat.custom_params, {'kept': 1, 'thinking_budget': 2048})
        sent = CONVERTED[0]
        self.assertEqual((sent.thinking.type, sent.thinking.budget_tokens), ('adaptive', None))
        self.assertEqual(sent.output_config.effort, 'high')

    def test_unbudgeted_requests_pass_through(self):
        for thinking in ({}, {'type': 'adaptive'}, {'type': 'disabled'}):
            original = request(**thinking)
            chat = AnthropicServing._convert_to_chat_completion_request(None, original)
            self.assertIs(CONVERTED[-1], original)
            self.assertEqual(chat.custom_params, {'kept': 1})


class GrammarPathTests(unittest.TestCase):
    def setUp(self):
        PROCESSED.clear()

    def test_plain_request_skips_grammar(self):
        self.assertFalse(GrammarManager.process_req_with_grammar(None, req()))
        self.assertFalse(GrammarManager.process_req_with_grammar(None, req(custom_params={'other': 1})))
        self.assertEqual(PROCESSED, [])

    def test_budgeted_and_constrained_requests_take_grammar(self):
        for params in ({'custom_params': {'thinking_budget': 0}}, {'custom_params': {'thinking_budget': 1024}},
                       {'json_schema': '{}'}, {'regex': 'a+'}, {'ebnf': 'root ::= "a"'}, {'structural_tag': '{}'}):
            self.assertTrue(GrammarManager.process_req_with_grammar(None, req(**params)), params)
        self.assertEqual(len(PROCESSED), 6)

    def test_invalid_budgets_skip_grammar(self):
        for budget in (-1, 1.5, True, '1024'):
            self.assertFalse(GrammarManager.process_req_with_grammar(
                None, req(custom_params={'thinking_budget': budget})), budget)
        self.assertEqual(PROCESSED, [])


if __name__ == '__main__':
    unittest.main()
