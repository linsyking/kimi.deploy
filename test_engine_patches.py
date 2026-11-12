"""kimictl's import-time engine patches: model sampling defaults and the medium effort tier.

Each case imports kimictl in a fresh interpreter (the venv's, as the engine does) with the
patch's environment, so it runs against whichever SGLang the venv holds; no GPU or server.

    CUDA_VISIBLE_DEVICES='' .venvs/<venv>/bin/python -m unittest test_engine_patches.py"""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parent
LOAD_KIMICTL = f'''
import importlib.machinery, importlib.util, json, types
loader = importlib.machinery.SourceFileLoader('kimictl_under_test', {str(ROOT / 'kimictl')!r})
module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(module)
'''
DEFAULTS = '''
from sglang.srt.configs.model_config import ModelConfig
def defaults(mode, generation):
    config = types.SimpleNamespace(sampling_defaults=mode, hf_generation_config=None if generation is None
                                   else types.SimpleNamespace(to_dict=lambda: generation))
    return ModelConfig.get_default_sampling_params(config)
print(json.dumps([defaults('model', None), defaults('model', {'top_p': 0.9}), defaults('openai', None)]))
'''
TIERS = '''
from sglang.srt.entrypoints.openai import serving_chat
tiers = getattr(serving_chat, '_KIMI_K31_EFFORT_TIERS', None)
if tiers is None:
    from sglang.srt.entrypoints.openai import serving_kimi_k3
    tiers = serving_kimi_k3._KIMI_K31_LEVEL_BY_TIER
print(json.dumps(tiers))
'''


def run(code, **env):
    keep = {key: value for key, value in os.environ.items() if not key.startswith('KIMI_')}
    result = subprocess.run([sys.executable, '-c', LOAD_KIMICTL + code], env={**keep, **env},
                            capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise AssertionError(result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


class DefaultSamplingTests(unittest.TestCase):
    def test_fills_what_generation_config_lacks(self):
        filled, generation, openai = run(DEFAULTS, KIMI_DEFAULT_SAMPLING=json.dumps({'temperature': 1.0, 'top_p': 0.95}))
        self.assertEqual(filled, {'temperature': 1.0, 'top_p': 0.95})
        self.assertEqual(generation, {'temperature': 1.0, 'top_p': 0.9})
        self.assertEqual(openai, {})

    def test_unset_leaves_sglang_alone(self):
        self.assertEqual(run(DEFAULTS), [{}, {'top_p': 0.9}, {}])


class MediumEffortTests(unittest.TestCase):
    def test_medium_maps_to_configured_tier(self):
        tiers = run(TIERS, KIMI_MEDIUM_EFFORT='high')
        self.assertEqual(tiers['medium'], 'high')
        self.assertEqual({tiers[name] for name in ('low', 'high', 'max')}, {'low', 'high', 'max'})


if __name__ == '__main__':
    unittest.main()
