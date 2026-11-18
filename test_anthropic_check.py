import unittest

from anthropic_check import citations_allowed


class CitationValidationTests(unittest.TestCase):
    def test_accepts_allowed_exact_and_subdomain_hosts(self):
        answer = ('[SGLang](https://docs.sglang.io/docs/basic_usage/anthropic_api) and '
                  '[source](https://github.com/sgl-project/sglang)')
        self.assertTrue(citations_allowed(answer, ('docs.sglang.io', 'github.com')))

    def test_rejects_domain_lookalikes(self):
        answer = '[not GitHub](https://github.com.evil.example/repository)'
        self.assertFalse(citations_allowed(answer, ('github.com',)))

    def test_rejects_one_disallowed_citation(self):
        answer = ('[allowed](https://github.com/sgl-project/sglang) and '
                  '[other](https://example.com/)')
        self.assertFalse(citations_allowed(answer, ('github.com',)))

    def test_requires_a_markdown_source_url(self):
        self.assertFalse(citations_allowed('See github.com/sgl-project/sglang', ('github.com',)))


if __name__ == '__main__':
    unittest.main()
