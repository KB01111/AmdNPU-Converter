"""Worker trust-boundary regressions; no model downloads or ML dependencies."""
import contextlib
import enum
import fnmatch
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

WORKERS = Path(__file__).resolve().parents[1]


def load(relative):
    path = WORKERS / relative
    spec = importlib.util.spec_from_file_location('_worker_test_' + relative.replace('/', '_').replace('.', '_'), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def invoke(module, args):
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = module.main(args)
    return code, json.loads(stdout.getvalue()) if stdout.getvalue().strip() else None, stderr.getvalue()


class Array:
    def __init__(self, values):
        self.values = list(values.values if isinstance(values, Array) else values)
        self.shape = (len(self.values),)
        self.size = len(self.values)

    def __sub__(self, other):
        return Array(a - b for a, b in zip(self.values, other.values))

    def all(self):
        return all(self.values)


NP = types.SimpleNamespace(
    asarray=lambda values, **kwargs: values if isinstance(values, Array) else Array(values),
    ones=lambda shape, **kwargs: Array([1] * math.prod(shape)),
    abs=lambda arr: Array(abs(x) for x in arr.values),
    mean=lambda arr: sum(arr.values) / len(arr.values),
    max=lambda arr: max(arr.values),
    isfinite=lambda arr: Array(math.isfinite(x) for x in arr.values),
    allclose=lambda a, b, **kwargs: all(math.isclose(x, y, rel_tol=kwargs['rtol'], abs_tol=kwargs['atol'])
                                      for x, y in zip(a.values, b.values)),
    float64='float64', float32='float32', int64='int64',
)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.module = load('evaluation/bench.py')

    def test_missing_accelerator_is_rejected(self):
        for device in ('npu', 'gpu', 'typo'):
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.module.providers_for(device, ['CPUExecutionProvider'])

    def test_generic_provider_does_not_select_npu(self):
        with self.assertRaises(ValueError):
            self.module.providers_for('npu', ['DmlExecutionProvider', 'OpenVINOExecutionProvider'])

    def test_explicit_cuda_wins_over_default_directml(self):
        self.assertEqual(self.module.providers_for('gpu',
            ['CPUExecutionProvider', 'DmlExecutionProvider', 'CUDAExecutionProvider'], 'cuda'),
            ['CUDAExecutionProvider'])

    def test_unavailable_explicit_provider_rejects_other_accelerators(self):
        for device, available, ep in [('gpu', ['CUDAExecutionProvider'], 'dml'),
                                      ('npu', ['NPUExecutionProvider'], 'vitisai')]:
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.module.providers_for(device, available, ep)

    def test_provider_pin_leaves_other_comparison_devices_at_defaults(self):
        self.assertEqual(self.module.providers_for('cpu',
            ['CPUExecutionProvider', 'CUDAExecutionProvider'], 'cuda'), ['CPUExecutionProvider'])

    def test_exact_provider_names_can_pin_gpu(self):
        self.assertEqual(self.module.providers_for('gpu',
            ['DmlExecutionProvider', 'CUDAExecutionProvider'], 'CUDAExecutionProvider'),
            ['CUDAExecutionProvider'])

    def benchmark(self, devices, available, actual=None, events=None, iterations='2', ep=None, ep_devices=None):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'model.onnx'
            model.write_bytes(b'model')
            class Options:
                def add_provider_for_devices(self, devices, options):
                    self.selected_devices = devices
            class Session:
                def __init__(self, path, providers=None, sess_options=None):
                    if ep_devices is not None:
                        if providers is not None or not getattr(sess_options, 'selected_devices', None):
                            raise ValueError('dynamic providers must bind device handles through SessionOptions')
                        providers = [sess_options.selected_devices[0].ep_name]
                    self.providers = actual or providers
                def get_inputs(self):
                    return [types.SimpleNamespace(name='x', shape=[1], type='tensor(float)')]
                def get_providers(self):
                    return self.providers
                def run(self, outputs, feed):
                    return [Array([1])]
                def end_profiling(self):
                    profile = Path(tmp) / 'profile.json'
                    recorded = events if events is not None else [
                        {'cat': 'Node', 'name': 'node_kernel_time', 'args': {'provider': self.providers[0]}}]
                    profile.write_text(json.dumps(recorded), encoding='utf-8')
                    return str(profile)
            ort = types.SimpleNamespace(InferenceSession=Session, SessionOptions=Options,
                                        get_available_providers=lambda: available)
            if ep_devices is not None:
                ort.get_ep_devices = lambda: ep_devices
            with patch.dict(sys.modules, {'numpy': NP, 'onnxruntime': ort, 'psutil': None, 'windowsml': None}):
                args = ['--model', str(model), '--devices', devices,
                        '--iterations', iterations, '--warmup', '0']
                if ep:
                    args += ['--ep', ep]
                return invoke(self.module, args)

    def test_dynamic_npu_provider_binds_the_npu_device_handle(self):
        handle = types.SimpleNamespace(ep_name='VitisAIExecutionProvider',
            device=types.SimpleNamespace(type=types.SimpleNamespace(name='NPU'), metadata={}), ep_metadata={})
        code, rows, error = self.benchmark('npu', ['CPUExecutionProvider'], ep_devices=[handle], ep='vitisai')
        self.assertEqual(code, 0, error)
        self.assertTrue(rows[0]['npu_verified'])

    def test_npu_provider_name_cannot_select_its_gpu_device(self):
        handle = types.SimpleNamespace(ep_name='VitisAIExecutionProvider',
            device=types.SimpleNamespace(type=types.SimpleNamespace(name='GPU'), metadata={}), ep_metadata={})
        code, rows, _ = self.benchmark('npu', ['CPUExecutionProvider'], ep_devices=[handle], ep='vitisai')
        self.assertNotEqual(code, 0)
        self.assertIn('no matching npu device', rows[0]['error'])

    def test_cli_pins_cuda_while_cpu_comparison_keeps_cpu(self):
        code, rows, _ = self.benchmark('cpu,gpu',
            ['CPUExecutionProvider', 'DmlExecutionProvider', 'CUDAExecutionProvider'], ep='cuda')
        self.assertEqual(code, 0)
        self.assertEqual([row['providers'][0] for row in rows],
                         ['CPUExecutionProvider', 'CUDAExecutionProvider'])

    def test_gpu_registration_cannot_hide_cpu_or_mixed_kernel_fallback(self):
        for providers in ([], ['CPUExecutionProvider'], ['CUDAExecutionProvider', 'CPUExecutionProvider']):
            events = [{'cat': 'Node', 'name': 'node_kernel_time', 'args': {'provider': p}} for p in providers]
            with self.subTest(providers=providers):
                code, rows, _ = self.benchmark('gpu', ['CUDAExecutionProvider'], events=events)
                self.assertNotEqual(code, 0)
                self.assertIn('error', rows[0])

    def test_mixed_results_preserved_with_nonzero_exit(self):
        code, rows, _ = self.benchmark('cpu,gpu', ['CPUExecutionProvider'])
        self.assertNotEqual(code, 0)
        self.assertIn('latency_ms_mean', rows[0])
        self.assertIn('error', rows[1])

    def test_actual_session_fallback_is_error(self):
        code, rows, _ = self.benchmark('npu', ['VitisAIExecutionProvider'], ['CPUExecutionProvider'])
        self.assertNotEqual(code, 0)
        self.assertIn('error', rows[0])

    def test_provider_name_without_placement_is_not_npu_verified(self):
        _, rows, _ = self.benchmark('npu', ['VitisAIExecutionProvider'], events=[])
        self.assertFalse(rows[0]['npu_verified'])

    def test_only_exclusive_npu_node_evidence_verifies(self):
        for providers, expected in [(['VitisAIExecutionProvider'], True),
                                    (['CPUExecutionProvider'], False),
                                    (['DmlExecutionProvider'], False),
                                    (['VitisAIExecutionProvider', 'CPUExecutionProvider'], False)]:
            events = [{'cat': 'Node', 'name': 'node_kernel_time', 'args': {'provider': p}} for p in providers]
            with self.subTest(providers=providers):
                _, rows, _ = self.benchmark('npu', ['VitisAIExecutionProvider'], events=events)
                self.assertEqual(rows[0]['npu_verified'], expected)

    def test_zero_iterations_rejected(self):
        code, _, _ = self.benchmark('cpu', ['CPUExecutionProvider'], iterations='0')
        self.assertNotEqual(code, 0)


class RuntimeDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.module = load('evaluation/bench.py')

    def test_installed_catalog_registration_exposes_dynamic_npu_without_download(self):
        self.assertTrue(hasattr(self.module, 'runtime_providers'), 'installed-only runtime discovery is missing')
        calls = []
        class ReadyState(enum.Enum):
            Ready = 0
            NotReady = 1
            NotPresent = 2
        class Provider:
            def __init__(self, name, state):
                self.name, self.ready_state = name, state
                self.library_path = name + '.dll'
            def ensure_ready(self):
                if self.ready_state == ReadyState.NotPresent:
                    raise AssertionError('must never acquire a missing provider')
                calls.append(('prepare', self.name))
                self.ready_state = ReadyState.Ready
        providers = [Provider('VitisAIExecutionProvider', ReadyState.NotReady),
                     Provider('QNNExecutionProvider', ReadyState.NotPresent)]
        class Catalog:
            def find_all_providers(self): return providers
            def close(self): calls.append(('close',))
        devices = [types.SimpleNamespace(ep_name='CPUExecutionProvider',
                    device=types.SimpleNamespace(type=types.SimpleNamespace(name='CPU'), metadata={}), ep_metadata={})]
        def register(name, library):
            calls.append(('register', name, library))
            devices.append(types.SimpleNamespace(ep_name=name,
                device=types.SimpleNamespace(type=types.SimpleNamespace(name='NPU'), metadata={'Description': 'AMD NPU'}), ep_metadata={}))
        ort = types.SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'],
            get_ep_devices=lambda: devices, register_execution_provider_library=register,
            unregister_execution_provider_library=lambda name: calls.append(('unregister', name)))
        with patch.dict(sys.modules, {'windowsml': types.SimpleNamespace(EpCatalog=Catalog)}), \
             patch.object(sys, 'platform', 'win32'):
            with self.module.runtime_providers(ort) as (report, handles):
                self.assertEqual(report['providers'], ['CPUExecutionProvider', 'VitisAIExecutionProvider'])
                self.assertIn({'provider': 'VitisAIExecutionProvider', 'device': 'npu', 'name': 'AMD NPU'}, report['devices'])
                self.assertEqual(len(handles), 2)
        self.assertEqual([call for call in calls if len(call) > 1 and call[1] == 'QNNExecutionProvider'], [])
        self.assertIn(('prepare', 'VitisAIExecutionProvider'), calls)
        self.assertIn(('register', 'VitisAIExecutionProvider', 'VitisAIExecutionProvider.dll'), calls)

    def test_discovery_does_not_require_model_or_numpy(self):
        ort = types.SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        with patch.dict(sys.modules, {'onnxruntime': ort, 'numpy': None, 'windowsml': None}):
            try:
                code, report, error = invoke(self.module, ['--discover-json'])
            except SystemExit as exc:
                self.fail(f'discovery CLI unavailable: {exc}')
        self.assertEqual(code, 0, error)
        self.assertEqual(report['providers'], ['CPUExecutionProvider'])
        self.assertIsInstance(report['devices'], list)
        self.assertIsInstance(report['errors'], list)
        self.assertFalse(report['windowsml_available'])

    def test_conflicting_ort_distributions_fail_before_import(self):
        self.assertTrue(hasattr(self.module, 'check_ort_distributions'), 'ORT namespace conflict check is missing')
        versions = {'onnxruntime': '1.25', 'onnxruntime-windowsml': '1.25'}
        def version(name):
            if name in versions:
                return versions[name]
            raise self.module.metadata.PackageNotFoundError(name)
        with patch.object(self.module.metadata, 'version', side_effect=version), self.assertRaisesRegex(ValueError, 'conflicting'):
            self.module.check_ort_distributions()


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.module = load('evaluation/compare.py')

    def compare(self, baseline, candidate, *, missing=False, threshold='0.02', supplied=True, candidate_name='output'):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ('base.onnx', 'candidate.onnx', 'eval.npz')]
            for path in paths[:2] + ([] if missing else paths[2:]):
                path.write_bytes(b'fixture')
            class Dataset(dict):
                files = ['x']
                def close(self): pass
                def __enter__(self): return self
                def __exit__(self, *args): pass
            np = types.SimpleNamespace(**vars(NP), load=lambda *args, **kwargs: Dataset(x=Array([1])))
            session = types.SimpleNamespace(get_inputs=lambda: [types.SimpleNamespace(name='x', shape=[1], type='tensor(float)')])
            ort = types.SimpleNamespace(InferenceSession=lambda *args, **kwargs: session)
            args = ['--baseline', str(paths[0]), '--candidate', str(paths[1]), '--max-degradation', threshold]
            if supplied:
                args += ['--eval-npz', str(paths[2])]
            with patch.dict(sys.modules, {'numpy': np, 'onnxruntime': ort}), \
                 patch.object(self.module, 'run_model', side_effect=[(baseline, [{'name': 'output'}]),
                                                                   (candidate, [{'name': candidate_name}])]):
                return invoke(self.module, args)

    def test_missing_supplied_dataset_rejected(self):
        code, _, _ = self.compare([Array([1])], [Array([1])], missing=True)
        self.assertNotEqual(code, 0)

    def test_missing_or_extra_outputs_rejected(self):
        for candidate in ([], [Array([1]), Array([1])]):
            code, report, _ = self.compare([Array([1])], candidate)
            self.assertNotEqual(code, 0)
            self.assertFalse(report and report.get('pass', False))

    def test_nonfinite_values_and_thresholds_rejected(self):
        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value):
                code, _, _ = self.compare([Array([value])], [Array([value])])
                self.assertNotEqual(code, 0)
        for threshold in ('nan', 'inf', '-1'):
            code, _, _ = self.compare([Array([1])], [Array([1])], threshold=threshold)
            self.assertNotEqual(code, 0)

    def test_output_names_must_match(self):
        code, _, _ = self.compare([Array([1])], [Array([1])], candidate_name='different')
        self.assertNotEqual(code, 0)

    def test_overflowed_reference_mean_cannot_hide_large_drift(self):
        code, _, _ = self.compare([Array([1e308, 1e308])], [Array([1e308, 0])])
        self.assertNotEqual(code, 0)

    def test_finite_matching_outputs_pass_on_supplied_dataset(self):
        code, report, _ = self.compare([Array([1, 2])], [Array([1, 2])])
        self.assertEqual(code, 0)
        self.assertTrue(report['pass'])

    def test_synthetic_input_is_not_accuracy_acceptance(self):
        code, report, _ = self.compare([Array([1])], [Array([1])], supplied=False)
        self.assertNotEqual(code, 0)
        self.assertFalse(report and report.get('pass', False))


class DownloadTests(unittest.TestCase):
    def test_cua_forms_pair_normalized_without_code_or_pickle(self):
        module = load('download.py')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'snapshot'
            def snapshot_download(**kwargs):
                for name in ('cua-s1-forms.json', 'cua-s1-forms.safetensors', 'cua-s1-forms.pt'):
                    if any(fnmatch.fnmatch(name, pat) for pat in kwargs['allow_patterns']):
                        out.mkdir(exist_ok=True)
                        (out / name).write_bytes(b'fixture')
            with patch.dict(sys.modules, {'huggingface_hub': types.SimpleNamespace(snapshot_download=snapshot_download)}):
                code, _, _ = invoke(module, ['--model', 'cua-ai/cua-s1-forms', '--out', str(out)])
            self.assertEqual(code, 0)
            self.assertTrue((out / 'config.json').is_file())
            self.assertTrue((out / 'model.safetensors').is_file())
            self.assertFalse(list(out.glob('*.pt')))

    def test_safe_snapshot_excludes_executable_and_pickle_files(self):
        path = WORKERS / 'download.py'
        self.assertTrue(path.is_file(), 'safe download worker is missing')
        module = load('download.py')
        files = ['config.json', 'model.safetensors', 'tokenizer.json', 'pytorch_model.bin', 'model.pt', 'custom.py']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'snapshot'
            def snapshot_download(**kwargs):
                for name in files:
                    if any(fnmatch.fnmatch(name, pat) for pat in kwargs['allow_patterns']):
                        out.mkdir(exist_ok=True)
                        (out / name).write_bytes(b'fixture')
                return str(out)
            hub = types.SimpleNamespace(snapshot_download=snapshot_download)
            with patch.dict(sys.modules, {'huggingface_hub': hub}):
                code, _, _ = invoke(module, ['--model', 'org/model', '--out', str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(sorted(p.name for p in out.iterdir() if p.is_file()),
                             ['config.json', 'model.safetensors', 'tokenizer.json'])


class PreparationTests(unittest.TestCase):
    def test_external_tensor_traversal_rejected_before_loading_data(self):
        module = load('prepare_onnx.py')
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source.onnx', Path(tmp) / 'output.onnx'
            source.write_bytes(b'source')
            model = types.SimpleNamespace(external_data=[types.SimpleNamespace(key='location', value='../secret')])
            onnx = types.SimpleNamespace(load=lambda *a, **k: model)
            with patch.dict(sys.modules, {'onnx': onnx}):
                code, _, error = invoke(module, ['--input', str(source), '--output', str(output)])
            self.assertNotEqual(code, 0)
            self.assertIn('within input directory', error)
            self.assertFalse(output.exists())

    def test_source_is_preserved_and_output_checked(self):
        self.assertTrue((WORKERS / 'prepare_onnx.py').is_file(), 'ONNX preparation worker is missing')
        module = load('prepare_onnx.py')
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'source.onnx', Path(tmp) / 'out.onnx'
            source.write_bytes(b'source')
            model = types.SimpleNamespace(opset_import=[types.SimpleNamespace(domain='', version=17)])
            checked = []
            onnx = types.SimpleNamespace(load=lambda *a, **k: model,
                checker=types.SimpleNamespace(check_model=lambda p: checked.append(p)),
                save_model=lambda m, p, **k: Path(p).write_bytes(b'prepared'),
                external_data_helper=types.SimpleNamespace(load_external_data_for_model=lambda *a: None))
            with patch.dict(sys.modules, {'onnx': onnx}):
                code, _, _ = invoke(module, ['--input', str(source), '--output', str(output), '--opset', '17'])
            self.assertEqual(code, 0)
            self.assertEqual(source.read_bytes(), b'source')
            self.assertEqual(output.read_bytes(), b'prepared')
            self.assertTrue(checked)


class ExportSafetyTests(unittest.TestCase):
    def test_cua_export_passes_official_forms_and_nano_batch_shapes(self):
        module = load('exporters/cua-s1/export_cua_s1.py')
        class Module:
            def eval(self): return self
            def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
        class Tensor:
            def __init__(self, shape): self.shape = shape
        for variant in ('forms', 'nano-text'):
            seen = []
            class Model(Module):
                def forward(self, batch):
                    required = {'context_ids', 'option_ids', 'option_token_mask', 'option_mask'}
                    required |= {'context_token_mask', 'modality'} if variant == 'nano-text' else {'context_mask'}
                    if set(batch) != required:
                        raise ValueError('incorrect official model batch')
                    seen.append(batch)
                    return Tensor((1, 4))
            loader = lambda *a, **k: (Model(), None, {'context_tokens': 224, 'option_tokens': 96})
            def export(model, inputs, path, **kwargs):
                model(*inputs)
                Path(path).write_bytes(b'onnx')
            torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=Module), no_grad=contextlib.nullcontext,
                ones=lambda *shape, **k: Tensor(shape), long='int64', bool='bool',
                backends=types.SimpleNamespace(), onnx=types.SimpleNamespace(export=export))
            modules = {'torch': torch, 'cua_s1': types.ModuleType('cua_s1'),
                       'cua_s1.model': types.SimpleNamespace(load_checkpoint=loader),
                       'cua_s1.nano': types.SimpleNamespace(load_nano_checkpoint=loader)}
            with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, modules), \
                 patch.object(module, 'validate_onnx', return_value={'ort_run': 'ok', 'numeric_compare': [{'allclose': True}]}):
                root = Path(tmp)
                for name in ('model.safetensors', 'config.json'):
                    (root / name).touch()
                code, report, error = invoke(module, ['--weights', str(root / 'model.safetensors'),
                    '--config', str(root / 'config.json'), '--output', str(root / 'model.onnx'), '--variant', variant])
            with self.subTest(variant=variant):
                self.assertEqual(code, 0, error)
                self.assertTrue(report['validation_passed'])
                self.assertEqual(seen[0]['context_ids'].shape, (1, 224))

    def test_cua_uses_official_signature_loader_and_propagates_failure(self):
        module = load('exporters/cua-s1/export_cua_s1.py')
        self.assertTrue(hasattr(module, 'load_official_checkpoint'), 'official safe checkpoint loader missing')
        def reject(*args, **kwargs):
            raise ValueError('checkpoint state signature mismatch')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('model.safetensors', 'config.json'):
                (root / name).touch()
            with patch.dict(sys.modules, {'cua_s1': types.ModuleType('cua_s1'),
                'cua_s1.model': types.SimpleNamespace(load_checkpoint=reject)}), self.assertRaisesRegex(ValueError, 'signature'):
                module.load_official_checkpoint(str(root / 'model.safetensors'), str(root / 'config.json'), 'forms')

    def test_common_validation_rejects_missing_and_nonfinite_outputs(self):
        module = load('exporters/common.py')
        for reference, outputs in [([Array([1]), Array([1])], [Array([1])]),
                                   ([Array([float('inf')])], [Array([float('inf')])])]:
            session = types.SimpleNamespace(get_inputs=lambda: [], run=lambda *args: outputs)
            ort = types.SimpleNamespace(InferenceSession=lambda *args, **kwargs: session)
            onnx = types.SimpleNamespace(checker=types.SimpleNamespace(check_model=lambda *args: None))
            with patch.dict(sys.modules, {'numpy': NP, 'onnxruntime': ort, 'onnx': onnx, 'torch': None}):
                report = module.validate_onnx('fixture.onnx', {}, torch_outputs=reference)
            comparison = report.get('numeric_compare')
            self.assertFalse(isinstance(comparison, list) and comparison and
                             all(item.get('allclose') is True for item in comparison))

    def test_image_export_uses_pixels_and_safetensors(self):
        module = load('exporters/transformers/export_hf.py')
        class Module:
            def eval(self): return self
            def __call__(self, *args, **kwargs): return self.forward(*args, **kwargs)
        class Tensor:
            shape = (1, 3, 224, 224)
            def dim(self): return 4
        pixels = Tensor()
        class Model(Module):
            def forward(self, **kwargs):
                if set(kwargs) != {'pixel_values'}:
                    raise ValueError('image classifier requires pixel_values')
                return types.SimpleNamespace(logits=pixels)
        def from_pretrained(*args, **kwargs):
            if kwargs.get('trust_remote_code') is not False or kwargs.get('use_safetensors') is not True:
                raise ValueError('unsafe model loader')
            return Model()
        loader = types.SimpleNamespace(from_pretrained=from_pretrained)
        processor = types.SimpleNamespace(size={'height': 224, 'width': 224})
        processor_type = type('Processor', (), {'size': processor.size, '__call__': lambda *a, **k: {'pixel_values': pixels}})
        tf = types.SimpleNamespace(AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k:
            types.SimpleNamespace(architectures=['ConvNextForImageClassification'], image_size=224, num_channels=3)),
            AutoModel=loader, AutoModelForImageClassification=loader, AutoModelForSequenceClassification=loader,
            AutoModelForTokenClassification=loader,
            AutoImageProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **k: processor_type()),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: (_ for _ in ()).throw(ValueError('not text'))))
        exported = []
        def export(model, example, path, **kwargs):
            exported.append(kwargs['input_names'])
            model(*example)
            Path(path).write_bytes(b'onnx')
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=Module), no_grad=contextlib.nullcontext,
            zeros=lambda *a, **k: pixels, ones=lambda *a, **k: pixels, float32='float32',
            onnx=types.SimpleNamespace(export=export))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'torch': torch, 'transformers': tf}), \
             patch.object(module, 'validate_onnx', return_value={'ort_run': 'ok', 'numeric_compare': [{'allclose': True}]}):
            code, _, _ = invoke(module, ['--model-id', 'org/convnext', '--output', str(Path(tmp) / 'model.onnx')])
        self.assertEqual(code, 0)
        self.assertEqual(exported, [['pixel_values']])

    def test_quark_uses_requested_preset_instead_of_default_qconfig(self):
        module = load('quark/quantize.py')
        class QConfig:
            @staticmethod
            def get_default_config(name):
                return {'preset': name}
        with patch.dict(sys.modules, {'quark': types.ModuleType('quark'), 'quark.onnx': types.SimpleNamespace(QConfig=QConfig)}):
            self.assertEqual(module.profile_to_qconfig('a16w8'), {'preset': 'A16W8'})

    def test_calibration_mapping_builds_one_complete_multi_input_sample(self):
        module = load('quark/quantize.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('ids.npy', 'mask.npy'):
                (root / name).touch()
            np = types.SimpleNamespace(**vars(NP), load=lambda *args, **kwargs: Array([1]))
            with patch.dict(sys.modules, {'numpy': np}):
                feeds = module.load_calibration(['ids', 'mask'], root,
                    {'ids': 'ids.npy', 'mask': 'mask.npy'}, 64)
            self.assertEqual(len(feeds), 1)
            self.assertEqual(set(feeds[0]), {'ids', 'mask'})


class InspectionTests(unittest.TestCase):
    def test_official_cua_sidecar_routes_to_cua_exporter(self):
        module = load('hf-inspector/inspect.py')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'cua-s1-forms.json'
            config.write_text('{"config":{"encoder":"tinyx","context_tokens":224,"option_tokens":96}}', encoding='utf-8')
            class Api:
                def __init__(self, **kwargs): pass
                def model_info(self, **kwargs): return types.SimpleNamespace(pipeline_tag='other')
                def list_repo_files(self, **kwargs): return ['cua-s1-forms.json', 'cua-s1-forms.safetensors']
            hub = types.SimpleNamespace(HfApi=Api, hf_hub_download=lambda **kwargs: str(config))
            errors = types.SimpleNamespace(HfHubHTTPError=RuntimeError, RepositoryNotFoundError=FileNotFoundError)
            with patch.dict(sys.modules, {'huggingface_hub': hub, 'huggingface_hub.utils': errors}):
                code, report, _ = invoke(module, ['--model-id', 'cua-ai/cua-s1-forms'])
        self.assertEqual(code, 0)
        self.assertEqual(report['export_strategy'], 'cua-s1')
        self.assertIn('context_ids', report['inputs_hint'])

    def test_inspection_never_downloads_weight_shards(self):
        module = load('hf-inspector/inspect.py')
        downloads = []
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text('{"architectures":["ConvNextForImageClassification"]}', encoding='utf-8')
            class Api:
                def __init__(self, **kwargs): pass
                def model_info(self, **kwargs): return types.SimpleNamespace()
                def list_repo_files(self, **kwargs): return ['config.json', 'model.safetensors']
            def download(**kwargs):
                downloads.append(kwargs['filename'])
                return str(config)
            hub = types.SimpleNamespace(HfApi=Api, hf_hub_download=download)
            errors = types.SimpleNamespace(HfHubHTTPError=RuntimeError, RepositoryNotFoundError=FileNotFoundError)
            with patch.dict(sys.modules, {'huggingface_hub': hub, 'huggingface_hub.utils': errors}):
                code, report, _ = invoke(module, ['--model-id', 'org/model'])
        self.assertEqual(code, 0)
        self.assertEqual(downloads, ['config.json'])
        self.assertIn('pixel_values', report['inputs_hint'])


class BootstrapTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'PowerShell bootstrap is Windows specific')
    def test_failed_base_install_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for env in ('.venv', '.venv-winml', '.venv-quark'):
                (root / env / 'Scripts').mkdir(parents=True)
                (root / env / 'Scripts' / 'python.exe').touch()
            for name in ('requirements-winml.txt', 'requirements-quark.txt'):
                (root / name).touch()
            (root / 'uv.cmd').write_text('@if "%~5"=="-e" (exit /b 1)\n@exit /b 0\n', encoding='ascii')
            result = subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                str(WORKERS / 'bootstrap.ps1'), '-WorkersDir', str(root)],
                env={**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH']}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
