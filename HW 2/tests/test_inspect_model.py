import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from src.inspect_model import (parameter_rows, group_table, activation_norms,
    forward_hooks, lora_params_formula, device_allocated_bytes,
    device_metric_source, PeakMemory, probe_mode, memory_profile)


class Inputs(dict):
    def to(self, device):
        return self


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        self.device = torch.device('cpu')
        self.fail = False

    def get_decoder(self):
        return self

    def forward(self, input_ids):
        value = torch.ones(1, input_ids.shape[1], 4)
        for layer in self.layers:
            value = layer(value)
            if self.fail:
                raise RuntimeError('forward failed')
        return value


class AnatomyTests(unittest.TestCase):
    def test_shared_and_independent_parameters(self):
        for tied in (True, False):
            with self.subTest(tied=tied):
                model = torch.nn.Module()
                model.embed_tokens = torch.nn.Embedding(7, 4)
                model.lm_head = torch.nn.Linear(4, 7, bias=False)
                if tied:
                    model.lm_head.weight = model.embed_tokens.weight
                rows = parameter_rows(model)
                table = group_table(rows)
                self.assertEqual(sum(x['params'] for x in table), sum(p.numel() for p in model.parameters()))
                head = next(x for x in table if x['group'] == 'lm_head')
                self.assertEqual(head['params'], 0 if tied else 28)
                self.assertEqual(head['shape'], '7×4')

    def test_lora_uses_asymmetric_actual_shapes(self):
        model = torch.nn.Module()
        model.q_proj = torch.nn.Linear(4, 8)
        model.v_proj = torch.nn.Linear(4, 2)
        model.other = torch.nn.Linear(20, 20)
        self.assertEqual(lora_params_formula(model, 3, ['q_proj', 'v_proj']), 54)

    def exercise_hooks(self, fail):
        model = TinyModel()
        model.train()
        model.layers[1].eval()
        original_flags = [m.training for m in model.modules()]
        foreign = model.layers[0].register_forward_hook(lambda *args: None)
        counts = [len(m._forward_hooks) for m in model.modules()]
        tokenizer = lambda *a, **kw: Inputs(input_ids=torch.ones(1, 5, dtype=torch.long))
        params = {'hooks': {'prompt': 'test'}}
        with patch('src.inspect_model.build_prompt', return_value='test'):
            if fail:
                model.fail = True
                with self.assertRaisesRegex(RuntimeError, 'forward failed'):
                    activation_norms(tokenizer, model, params)
            else:
                first = activation_norms(tokenizer, model, params)
                second = activation_norms(tokenizer, model, params)
                self.assertEqual(first, second)
                self.assertEqual(first['n_tokens'], 5)
                self.assertEqual(set(first['layers'].values()), {0, 1, 2})
        self.assertEqual([len(m._forward_hooks) for m in model.modules()], counts)
        self.assertEqual([m.training for m in model.modules()], original_flags)
        foreign.remove()

    def test_hooks_repeat_and_preserve_foreign_hooks_and_modes(self):
        self.exercise_hooks(False)

    def test_hooks_clean_up_after_forward_exception(self):
        self.exercise_hooks(True)

    def test_partial_hook_registration_is_cleaned(self):
        good = torch.nn.Linear(1, 1)
        bad = SimpleNamespace(register_forward_hook=lambda fn: (_ for _ in ()).throw(RuntimeError('registration')))
        with self.assertRaisesRegex(RuntimeError, 'registration'):
            with forward_hooks({'good': good, 'bad': bad}):
                pass
        self.assertEqual(len(good._forward_hooks), 0)

    def test_mps_reads_driver_not_rss(self):
        with patch('torch.mps.driver_allocated_memory', return_value=123), patch('src.inspect_model.peak_rss', side_effect=AssertionError('RSS')):
            self.assertEqual(device_allocated_bytes(torch.device('mps')), 123)
            self.assertEqual(device_metric_source(torch.device('mps')), 'torch.mps.driver_allocated_memory')

    def test_mps_retains_intermediate_peak_after_memory_drops(self):
        with patch('src.inspect_model.synchronize'), patch('src.inspect_model.device_allocated_bytes', side_effect=[10, 100, 20, 5]):
            with PeakMemory(torch.device('mps'), interval=1000) as peak:
                peak.checkpoint('forward')
                peak.checkpoint('backward')
            self.assertEqual(peak.used, 100)
            self.assertFalse(peak.thread.is_alive())

    def test_sampler_stops_on_exception(self):
        with patch('src.inspect_model.synchronize'), patch('src.inspect_model.device_allocated_bytes', return_value=10):
            with self.assertRaisesRegex(RuntimeError, 'body'):
                with PeakMemory(torch.device('mps'), interval=1000) as peak:
                    raise RuntimeError('body')
            self.assertFalse(peak.thread.is_alive())

    def test_cuda_uses_allocator_high_water_mark(self):
        with patch('src.inspect_model.synchronize'), patch('src.inspect_model.device_allocated_bytes', return_value=10), patch('torch.cuda.reset_peak_memory_stats') as reset, patch('torch.cuda.max_memory_allocated', return_value=300):
            with PeakMemory(torch.device('cuda')) as peak:
                pass
            reset.assert_called_once()
            self.assertEqual(peak.used, 300)
            self.assertEqual(device_metric_source(torch.device('cuda')), 'torch.cuda.max_memory_allocated')

    def test_cpu_uses_rss(self):
        with patch('src.inspect_model.peak_rss', return_value=(1048576, 'ru_maxrss')):
            with PeakMemory(torch.device('cpu')) as peak:
                pass
            self.assertEqual(peak.result()['peak_mb'], 1)
            self.assertIsNone(peak.result()['peak_device_mb'])

    def test_probe_passes_params_and_propagates_failures(self):
        params = {'memory': {'seq_len': 123}}
        good = SimpleNamespace(returncode=0, stdout='log\n'+json.dumps({'mode':'lora','peak_mb':1}), stderr='')
        with patch('src.inspect_model.subprocess.run', return_value=good) as run:
            self.assertEqual(probe_mode('lora', params)['peak_mb'], 1)
            self.assertEqual(json.loads(run.call_args.kwargs['input']), params)
            self.assertIn('--params-stdin', run.call_args.args[0])
        bad = SimpleNamespace(returncode=7, stdout='', stderr='failed allocation')
        with patch('src.inspect_model.subprocess.run', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, 'failed allocation'):
                probe_mode('lora', params)
        for stdout in ('', '{}', 'not json'):
            with patch('src.inspect_model.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=stdout, stderr='')):
                with self.assertRaisesRegex(RuntimeError, 'Некорректный'):
                    probe_mode('lora', params)

    def test_every_repeat_gets_separate_probe_and_maximum(self):
        calls = []
        def probe(mode, params):
            calls.append(mode)
            return {'mode':mode, 'peak_mb':len(calls)}
        with patch('src.inspect_model.probe_mode', side_effect=probe):
            result = memory_profile({'memory': {'repeats':2}})
        self.assertEqual(calls, ['inference','inference','full_ft','full_ft','lora','lora'])
        self.assertEqual([x['peak_mb'] for x in result], [2,4,6])
        self.assertTrue(all(len(x['runs']) == 2 for x in result))


if __name__ == '__main__':
    unittest.main()
