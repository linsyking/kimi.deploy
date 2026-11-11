"""CPU-only checks of checkpoint_overlay.py and modelctl on a small synthetic checkpoint and patch.

    CUDA_VISIBLE_DEVICES= .venvs/sglang/bin/python -m unittest -v test_checkpoint_overlay.py"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
import zstandard

import checkpoint_overlay as overlay

loader = importlib.machinery.SourceFileLoader('modelctl_tests', str(Path(__file__).with_name('modelctl')))
spec = importlib.util.spec_from_loader(loader.name, loader)
modelctl = importlib.util.module_from_spec(spec)
loader.exec_module(modelctl)

SHARDS = {'model-00001-of-000002.safetensors': ['l0.self_attn.o_proj.weight', 'l0.experts.weight_packed'],
          'model-00002-of-000002.safetensors': ['l1.mlp.down_proj.weight', 'l1.routed_expert_up_proj.weight']}
PATCHED = ['l0.self_attn.o_proj.weight', 'l1.mlp.down_proj.weight', 'l1.routed_expert_up_proj.weight']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def raw(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


class Fixture(unittest.TestCase):
    """A two-shard official checkpoint and a pqhaz-format XOR patch of three of its tensors."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.base, self.source = root / 'base', root / 'source'
        self.overlay = root / 'out/overlay.safetensors'
        (self.source / 'patch').mkdir(parents=True)
        self.base.mkdir()
        generator = torch.Generator().manual_seed(0)
        shapes = {'l0.self_attn.o_proj.weight': (8, 12), 'l1.mlp.down_proj.weight': (8, 6),
                  'l1.routed_expert_up_proj.weight': (8, 4)}
        self.official, weight_map = {}, {}
        for shard, names in SHARDS.items():
            tensors = {name: (torch.randn(shapes[name], generator=generator).to(torch.bfloat16) if name in shapes
                              else torch.randint(0, 255, (16,), dtype=torch.uint8, generator=generator))
                       for name in names}
            save_file(tensors, self.base / shard)
            self.official.update(tensors)
            weight_map.update(dict.fromkeys(names, shard))
        index = json.dumps({'metadata': {}, 'weight_map': weight_map}).encode()
        (self.base / 'model.safetensors.index.json').write_bytes(index)
        (self.base / '.kimi-verified.json').write_text(json.dumps({'revision': overlay.BASE_REVISION}))
        self.patched = {name: torch.randn(shapes[name], generator=generator).to(torch.bfloat16) for name in PATCHED}
        changes = []
        for number, name in enumerate(PATCHED):
            shard = weight_map[name]
            header, start = overlay.safetensors_header(self.base / shard)
            begin, end = header[name]['data_offsets']
            before, after = (self.base / shard).read_bytes()[start + begin:start + end], raw(self.patched[name])
            compressed = zstandard.ZstdCompressor().compress(bytes(a ^ b for a, b in zip(before, after)))
            (self.source / f'patch/{number:04d}.xor.zst').write_bytes(compressed)
            changes.append({'name': name, 'shape': list(shapes[name]), 'dtype': 'BF16', 'bytes': len(before),
                            'shard': shard, 'offset': start + begin, 'before_sha256': sha(before),
                            'after_sha256': sha(after), 'xor_file': f'{number:04d}.xor.zst',
                            'xor_sha256': sha(compressed), 'compressed_bytes': len(compressed)})
        manifest = json.dumps({'index_sha256': sha(index), 'changes': changes}).encode()
        (self.source / 'patch/manifest.json').write_bytes(manifest)
        pin = patch.object(overlay, 'MANIFEST_SHA256', sha(manifest))
        pin.start()
        self.addCleanup(pin.stop)

    def build(self):
        overlay.build(self.base, self.source, self.overlay)


class BuildTests(Fixture):
    def test_overlay_holds_exactly_the_patched_tensors(self):
        shards = {shard: (self.base / shard).read_bytes() for shard in SHARDS}
        self.build()
        built = load_file(self.overlay)
        self.assertEqual(sorted(built), sorted(PATCHED))
        for name in PATCHED:
            self.assertEqual(raw(built[name]), raw(self.patched[name]))
        self.assertEqual({shard: (self.base / shard).read_bytes() for shard in SHARDS}, shards)
        self.assertEqual(overlay.check(self.base, self.overlay)['tensor_names'], PATCHED)
        self.assertEqual(self.overlay.stat().st_mode & 0o777, 0o444)

    def test_modified_official_tensor_is_rejected(self):
        shard = self.base / 'model-00002-of-000002.safetensors'
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, 'does not match the manifest'):
            self.build()
        self.assertFalse(self.overlay.exists())

    def test_corrupt_overlay_fails_verify_and_check(self):
        self.build()
        time.sleep(0.05)  # file times advance by clock ticks, not nanoseconds
        self.overlay.chmod(0o644)
        data = bytearray(self.overlay.read_bytes())
        data[-1] ^= 1
        self.overlay.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, 'changed since it was verified'):
            overlay.check(self.base, self.overlay)
        with self.assertRaisesRegex(ValueError, '1 overlay tensors do not match'):
            overlay.verify(self.base, self.overlay)
        with self.assertRaisesRegex(ValueError, 'not verified on this node'):
            overlay.check(self.base, self.overlay)

    def test_touched_overlay_needs_verify_again(self):
        self.build()
        os.utime(self.overlay, ns=(1, 1))
        with self.assertRaisesRegex(ValueError, 'changed since it was verified'):
            overlay.check(self.base, self.overlay)
        overlay.verify(self.base, self.overlay)
        overlay.check(self.base, self.overlay)

    def test_unpinned_manifest_and_other_revision_are_rejected(self):
        self.build()
        with patch.object(overlay, 'MANIFEST_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'pinned'):
            overlay.check(self.base, self.overlay)
        (self.base / '.kimi-verified.json').write_text(json.dumps({'revision': '6d0a0202'}))
        with self.assertRaisesRegex(ValueError, 'verified official revision'):
            overlay.check(self.base, self.overlay)


class ReplacementTests(Fixture):
    def setUp(self):
        super().setUp()
        self.build()
        self.names = set(PATCHED)

    def test_replaces_only_patched_tensors_and_keeps_order(self):
        weights = [(name, self.official[name]) for names in SHARDS.values() for name in names]
        seen = set()
        result = list(overlay.replaced(weights, self.overlay, self.names, seen))
        self.assertEqual([name for name, _ in result], [name for name, _ in weights])
        self.assertEqual(seen, self.names)
        for (name, before), (_, after) in zip(weights, result):
            if name in self.names:
                self.assertEqual(raw(after), raw(self.patched[name]))
            else:
                self.assertIs(after, before)

    def test_rejects_shape_mismatch_and_duplicates(self):
        name = PATCHED[0]
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            list(overlay.replaced([(name, torch.zeros(3, dtype=torch.bfloat16))], self.overlay, self.names, set()))
        weight = self.official[name]
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            list(overlay.replaced([(name, weight), (name, weight)], self.overlay, self.names, set()))


class InstallTests(Fixture):
    """install() on SGLang's real DefaultModelLoader, its original replaced by a recorder."""

    def setUp(self):
        super().setUp()
        self.build()
        from sglang.srt.model_loader.loader import DefaultModelLoader
        saved = DefaultModelLoader.__dict__['load_weights_only']
        self.addCleanup(setattr, DefaultModelLoader, 'load_weights_only', saved)
        self.loaded = []
        DefaultModelLoader.load_weights_only = staticmethod(
            lambda model, weights, target_device: self.loaded.append(list(weights)))
        self.audit = self.overlay.parent / 'loads'
        overlay.install(str(self.overlay), str(self.base), str(self.audit))
        overlay.install(str(self.overlay), str(self.base), str(self.audit))  # idempotent
        self.load = DefaultModelLoader.load_weights_only
        self.weights = [(name, self.official[name]) for names in SHARDS.values() for name in names]

    def records(self):
        return [json.loads(path.read_text()) for path in self.audit.glob('rank-*.json')]

    def test_target_model_loads_the_overlay_and_records_it(self):
        self.load(type('KimiK31ForConditionalGeneration', (), {})(), iter(self.weights), None)
        loaded = dict(self.loaded[0])
        for name in PATCHED:
            self.assertEqual(raw(loaded[name]), raw(self.patched[name]))
        self.assertIs(loaded['l0.experts.weight_packed'], self.official['l0.experts.weight_packed'])
        [record] = self.records()
        self.assertEqual((record['rank'], record['tensor_count'], record['manifest_sha256'], record['overlay']),
                         (0, 3, overlay.MANIFEST_SHA256, str(self.overlay.resolve())))

    def test_draft_model_is_untouched(self):
        self.load(type('DFlash2DraftModel', (), {})(), iter(self.weights), None)
        self.assertTrue(all(after is before for (_, after), (_, before) in zip(self.loaded[0], self.weights)))
        self.assertEqual(self.records(), [])

    def test_failed_check_loads_official_weights_and_records_the_failure(self):
        self.overlay.with_suffix('.verified.json').unlink()
        self.load(type('KimiK31ForConditionalGeneration', (), {})(), iter(self.weights), None)
        self.assertTrue(all(after is before for (_, after), (_, before) in zip(self.loaded[0], self.weights)))
        [record] = self.records()
        self.assertEqual(record['tensor_count'], 0)
        self.assertIn('not verified', record['error'])


class SwitchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / '.runs').mkdir()
        self.original = b'{\n  "model_path": "official-shards",\n  "served_model_name": "kimi-k3.1-pro"\n}\n'
        (self.root / 'serving.json').write_bytes(self.original)
        checked = patch.object(modelctl, 'check_overlay')
        checked.start()
        self.addCleanup(checked.stop)

    def test_startup_failure_restores_exact_original_configuration(self):
        actions = []

        def control(root, *action):
            actions.append(action[0])
            if actions == ['stop', 'start']:
                raise subprocess.CalledProcessError(1, ['start'])
        with patch.object(modelctl, 'controller', control), patch.object(modelctl, 'validate_loaded'):
            with self.assertRaisesRegex(RuntimeError, 'previous service restored'):
                modelctl.switch(self.root, 'abliterated')
        self.assertEqual(actions, ['stop', 'start', 'stop', 'start'])
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_failed_overlay_audit_also_rolls_back(self):
        with patch.object(modelctl, 'controller') as controller, patch.object(
                modelctl, 'validate_loaded', side_effect=[subprocess.CalledProcessError(1, ['weights']), None]):
            with self.assertRaisesRegex(RuntimeError, 'previous service restored'):
                modelctl.switch(self.root, 'abliterated')
        self.assertEqual([call.args[1] for call in controller.call_args_list], ['stop', 'start', 'stop', 'start'])
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_interrupt_during_stop_restores_service(self):
        actions = []

        def control(root, *action):
            actions.append(action[0])
            if len(actions) == 1:
                raise KeyboardInterrupt('stop interrupted')
        with patch.object(modelctl, 'controller', control), patch.object(modelctl, 'validate_loaded'):
            with self.assertRaisesRegex(RuntimeError, 'previous service restored'):
                modelctl.switch(self.root, 'abliterated')
        self.assertEqual(actions, ['stop', 'stop', 'start'])
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_failed_rollback_stop_is_not_forced(self):
        actions = []

        def control(root, *action):
            actions.append(action)
            if len(actions) > 1:
                raise subprocess.CalledProcessError(1, list(action))
        with patch.object(modelctl, 'controller', control), patch.object(modelctl, 'validate_loaded'):
            with self.assertRaisesRegex(RuntimeError, 'Rollback failed'):
                modelctl.switch(self.root, 'abliterated')
        self.assertEqual(actions, [('stop',), ('start',), ('stop',)])

    def test_unverified_overlay_does_not_stop_service(self):
        with patch.object(modelctl, 'check_overlay', side_effect=ValueError('c13: not verified')), patch.object(
                modelctl, 'controller') as controller:
            with self.assertRaisesRegex(ValueError, 'not verified'):
                modelctl.switch(self.root, 'abliterated')
            controller.assert_not_called()
        self.assertEqual((self.root / 'serving.json').read_bytes(), self.original)

    def test_switches_change_only_the_overlay_selection(self):
        with patch.object(modelctl, 'controller'), patch.object(modelctl, 'validate_loaded'):
            modelctl.switch(self.root, 'abliterated')
            current = json.loads((self.root / 'serving.json').read_text())
            self.assertEqual(current.pop('checkpoint_overlay'), str(overlay.OVERLAY))
            self.assertEqual(current, json.loads(self.original))
            modelctl.switch(self.root, 'official')
        self.assertEqual(json.loads((self.root / 'serving.json').read_text()), json.loads(self.original))


if __name__ == '__main__':
    unittest.main()
