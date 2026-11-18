import asyncio
import copy
import json
import unittest

from web_search import WebSearchMiddleware


def request(tool_type='web_search_20250305'):
    return {'model': 'test', 'messages': [{'role': 'user', 'content':
            'Perform a web search for the query: SGLang OpenSERP integration'}],
            'tools': [{'type': tool_type, 'name': 'web_search'}],
            'tool_choice': {'type': 'tool', 'name': 'web_search'}}


class Searcher:
    def __init__(self):
        self.calls = []
        self.closed = False
        self.failure = False

    async def __call__(self, query, include, exclude):
        self.calls.append((query, include, exclude))
        await asyncio.sleep(0)
        if self.failure:
            raise RuntimeError('Provider error text must not be returned')
        return {'results': [{'title': 'SGLang', 'url': 'https://docs.sglang.ai/',
                             'highlights': ['Native search integration.']}]}

    async def close(self):
        self.closed = True


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.searcher = Searcher()
        self.forwarded = []
        async def app(scope, receive, send):
            if scope['type'] == 'lifespan':
                return
            first = await receive()
            self.forwarded.append((scope, first['body']))
            # A streaming disconnect listener must get the original receive
            # after the one replayed body, rather than replaying forever.
            self.assertEqual((await receive())['type'], 'http.disconnect')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'data: first\n\n', 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'data: done\n\n'})
        self.middleware = WebSearchMiddleware(app, searcher=self.searcher, api_key='test-key')

    async def invoke(self, body, auth=b'Bearer test-key', path='/v1/messages', middleware=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        pieces = [{'type': 'http.request', 'body': raw[:12], 'more_body': True},
                  {'type': 'http.request', 'body': raw[12:], 'more_body': False},
                  {'type': 'http.disconnect'}]
        async def receive():
            return pieces.pop(0)
        # The passthrough paths let the native app read the original stream.
        async def collect(scope, receive, send):
            if scope['type'] != 'http':
                return
            chunks = []
            while True:
                message = await receive()
                chunks.append(message.get('body', b''))
                if not message.get('more_body', False):
                    break
            self.forwarded.append((scope, b''.join(chunks)))
            self.assertEqual((await receive())['type'], 'http.disconnect')
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'data: first\n\n', 'more_body': True})
            await send({'type': 'http.response.body', 'body': b'data: done\n\n'})
        selected = middleware or self.middleware
        selected.app = collect
        sent = []
        async def send(message):
            sent.append(message)
        scope = {'type': 'http', 'method': 'POST', 'path': path,
                 'headers': [(b'authorization', auth), (b'content-length', str(len(raw)).encode())]}
        await selected(scope, receive, send)
        return sent

    async def test_search_rewrites_and_preserves_streaming(self):
        sent = await self.invoke(request())
        self.assertEqual(self.searcher.calls, [('SGLang OpenSERP integration', (), ())])
        scope, raw = self.forwarded[-1]
        result = json.loads(raw)
        self.assertNotIn('tools', result)
        self.assertNotIn('tool_choice', result)
        self.assertIn('https://docs.sglang.ai/', result['messages'][-1]['content'][-1]['text'])
        self.assertEqual(int(dict(scope['headers'])[b'content-length']), len(raw))
        self.assertEqual([m['body'] for m in sent[1:]], [b'data: first\n\n', b'data: done\n\n'])

    async def test_preserves_image_and_custom_tool(self):
        body = request()
        image = {'type': 'image', 'source': {'type': 'base64', 'data': 'abc', 'media_type': 'image/png'}}
        body['messages'][-1]['content'] = [image, {'type': 'text', 'text': 'SGLang search'}]
        custom = {'name': 'Read', 'input_schema': {'type': 'object'}}
        body['tools'].append(custom)
        await self.invoke(body)
        result = json.loads(self.forwarded[-1][1])
        self.assertEqual(result['messages'][-1]['content'][0], image)
        self.assertEqual(result['tools'], [custom])
        self.assertEqual(result['tool_choice'], {'type': 'auto'})

    async def test_non_search_is_byte_identical(self):
        raw = b'{ "messages": [{"role":"user","content":"hello"}], "tools": [] }'
        await self.invoke(raw)
        self.assertEqual(self.forwarded[-1][1], raw)
        self.assertFalse(self.searcher.calls)

    async def test_custom_websearch_is_not_a_server_tool(self):
        body = request()
        body['tools'] = [{'name': 'WebSearch', 'input_schema': {'type': 'object'}}]
        await self.invoke(body)
        self.assertFalse(self.searcher.calls)

    async def test_unauthorized_never_searches(self):
        for auth in (b'', b'Bearer wrong', b'\xff'):
            await self.invoke(request(), auth=auth)
        self.assertFalse(self.searcher.calls)

    async def test_other_endpoint_never_searches(self):
        await self.invoke(request(), path='/v1/chat/completions')
        self.assertFalse(self.searcher.calls)

    async def test_filters_and_cache(self):
        body = request()
        body['tools'][0]['allowed_domains'] = ['docs.sglang.ai']
        await self.invoke(body)
        await self.invoke(body)
        self.assertEqual(self.searcher.calls, [('SGLang OpenSERP integration', ('docs.sglang.ai',), ())])
        body['tools'][0].pop('allowed_domains')
        body['tools'][0]['blocked_domains'] = ['example.com']
        await self.invoke(body)
        self.assertEqual(len(self.searcher.calls), 2)
        self.assertEqual(self.searcher.calls[-1][2], ('example.com',))

    async def test_conflicting_filters_do_not_spend_credits(self):
        body = request()
        body['tools'][0].update(allowed_domains=['a.com'], blocked_domains=['b.com'])
        sent = await self.invoke(body)
        self.assertEqual(sent[0]['status'], 400)
        self.assertFalse(self.searcher.calls)
        self.assertFalse(self.forwarded)

    async def test_provider_failure_is_explicit_and_not_cached(self):
        self.searcher.failure = True
        sent = await self.invoke(request())
        self.assertEqual(sent[0]['status'], 502)
        self.assertNotIn(b'Provider error text', sent[-1]['body'])
        self.assertFalse(self.forwarded)
        self.searcher.failure = False
        await self.invoke(request())
        self.assertEqual(len(self.searcher.calls), 2)

    async def test_concurrent_identical_search_is_coalesced(self):
        await asyncio.gather(*(self.invoke(copy.deepcopy(request())) for _ in range(4)))
        self.assertEqual(len(self.searcher.calls), 1)

    async def test_search_disabled_by_request(self):
        body = request()
        body['tools'][0]['max_uses'] = 0
        await self.invoke(body)
        body = request()
        body['tool_choice'] = {'type': 'none'}
        await self.invoke(body)
        self.assertFalse(self.searcher.calls)

    async def test_versioned_and_unversioned_tools(self):
        for tool_type in ('web_search', 'web_search_20260209', 'web_search_20260318'):
            body = request(tool_type)
            await self.invoke(body)
            self.assertNotIn('tools', json.loads(self.forwarded[-1][1]))

    async def test_malformed_input_passes_to_native_validation(self):
        for raw in (b'not JSON', b'[]'):
            await self.invoke(raw)
            self.assertEqual(self.forwarded[-1][1], raw)
        body = request()
        body['tool_choice'] = 'invalid'
        await self.invoke(body)
        self.assertFalse(self.searcher.calls)

    async def test_lifespan_closes_provider_connection(self):
        async def ignore(*args):
            pass
        await self.middleware({'type': 'lifespan'}, ignore, ignore)
        self.assertTrue(self.searcher.closed)


class OpenSerpSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_fallback_payload_and_auth(self):
        import httpx
        from web_search import OpenSerpSearch
        calls = []

        def handler(request):
            engine = request.url.params['engines']
            calls.append(request)
            if engine == 'duckduckgo':
                return httpx.Response(503, json={'error': 'provider text must not escape'})
            return httpx.Response(200, json={'results': [{
                'type': 'organic', 'title': 'SGLang', 'url': 'https://github.com/sgl-project/sglang',
                'snippet': 'SGLang inference engine.',
            }]})

        search = OpenSerpSearch('https://search.test', 'test-key',
                                transport=httpx.MockTransport(handler))
        try:
            response = await search('SGLang OpenSERP integration', (), ())
        finally:
            await search.close()
        self.assertEqual([call.url.params['engines'] for call in calls], ['duckduckgo', 'bing'])
        request = calls[0]
        self.assertEqual(request.headers['authorization'], 'Bearer test-key')
        self.assertEqual(request.url.path, '/mega/search')
        self.assertEqual(request.url.params['mode'], 'balanced')
        self.assertEqual(request.url.params['limit'], '10')
        self.assertEqual(request.url.params['text'], 'SGLang OpenSERP integration')
        self.assertEqual(response['results'], [{
            'title': 'SGLang', 'url': 'https://github.com/sgl-project/sglang',
            'highlights': ['SGLang inference engine.'],
        }])

    async def test_domain_filters_and_result_cap(self):
        import httpx
        from web_search import OpenSerpSearch

        def handler(request):
            urls = ['https://docs.github.com/one', 'https://evil.github.com/skip',
                    'https://example.com/skip', 'https://github.com/two', 'https://github.com/three',
                    'https://github.com/four', 'https://github.com/five', 'https://github.com/six']
            return httpx.Response(200, json={'results': [
                {'title': url, 'url': url, 'snippet': 'evidence'} for url in urls
            ]})

        search = OpenSerpSearch('https://search.test', 'test-key',
                                transport=httpx.MockTransport(handler))
        try:
            response = await search('query', ('GitHub.com',), ('evil.github.com',))
        finally:
            await search.close()
        self.assertEqual([item['url'] for item in response['results']], [
            'https://docs.github.com/one', 'https://github.com/two', 'https://github.com/three',
            'https://github.com/four', 'https://github.com/five',
        ])

    async def test_empty_results_fall_back(self):
        import httpx
        from web_search import OpenSerpSearch
        calls = []

        def handler(request):
            engine = request.url.params['engines']
            calls.append(engine)
            results = [] if engine != 'yandex' else [{
                'title': 'Yandex result', 'url': 'https://example.org/', 'snippet': 'found',
            }]
            return httpx.Response(200, json={'results': results})

        search = OpenSerpSearch('https://search.test', 'test-key',
                                transport=httpx.MockTransport(handler))
        try:
            response = await search('query', (), ())
        finally:
            await search.close()
        self.assertEqual(calls, ['duckduckgo', 'bing', 'yandex'])
        self.assertEqual(response['results'][0]['title'], 'Yandex result')

    async def test_valid_empty_results_do_not_fail(self):
        import httpx
        from web_search import OpenSerpSearch

        def handler(request):
            return httpx.Response(200, json={'results': []})

        search = OpenSerpSearch('https://search.test', 'test-key',
                                transport=httpx.MockTransport(handler))
        try:
            response = await search('query', (), ())
        finally:
            await search.close()
        self.assertEqual(response, {'results': []})

    async def test_slow_response_is_bounded(self):
        import httpx
        from web_search import OpenSerpSearch
        calls = []

        async def handler(request):
            calls.append(request.url.params['engines'])
            await asyncio.sleep(0.2)
            return httpx.Response(200, json={'results': []})

        search = OpenSerpSearch('https://search.test', 'test-key', engines=('duckduckgo', 'bing', 'yandex', 'ecosia'),
                                timeout=0.05, engine_timeout=0.02, transport=httpx.MockTransport(handler))
        try:
            with self.assertRaisesRegex(RuntimeError, 'All OpenSERP engines failed'):
                await asyncio.wait_for(search('query', (), ()), timeout=0.5)
        finally:
            await search.close()
        self.assertTrue(calls)
        self.assertLessEqual(len(calls), 3)

    async def test_all_engines_fail(self):
        import httpx
        from web_search import OpenSerpSearch
        calls = []

        def handler(request):
            calls.append(request.url.params['engines'])
            return httpx.Response(503, json={'error': 'unavailable'})

        search = OpenSerpSearch('https://search.test', 'test-key',
                                transport=httpx.MockTransport(handler))
        try:
            with self.assertRaisesRegex(RuntimeError, 'All OpenSERP engines failed'):
                await search('query', (), ())
        finally:
            await search.close()
        self.assertEqual(calls, ['duckduckgo', 'bing', 'yandex'])


if __name__ == '__main__':
    unittest.main()
