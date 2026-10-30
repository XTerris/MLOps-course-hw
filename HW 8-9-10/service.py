"""FastAPI inference service; --backend stub is for process tests only."""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
import anyio


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest(manifest, root):
    if hashlib.sha256(canonical(manifest['runtime'])).hexdigest() != manifest['runtime_sha256']:
        raise ValueError('runtime_sha256 не соответствует runtime')
    files = manifest.get('file_hashes')
    if not isinstance(files, dict) or not files:
        raise ValueError('Не указан список проверяемых файлов артефакта')
    if hashlib.sha256(canonical(files)).hexdigest() != manifest['model_sha256']:
        raise ValueError('model_sha256 не соответствует списку файлов')
    uri = manifest.get('artifact_uri', '')
    artifact = Path(uri.removeprefix('file://'))
    if not artifact.is_absolute():
        artifact = root / artifact
    directory = artifact if artifact.is_dir() else artifact.parent
    for name, expected in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not name:
            raise ValueError('Недопустимый путь файла артефакта')
        if sha256(directory / name) != expected:
            raise ValueError(f'Несовпадение SHA-256 файла {name}')
    return artifact


class StubBackend:
    def __init__(self, manifest, artifact=None):
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0

    def warmup(self):
        pass

    def validate_prompt(self, prompt):
        return None

    def generate(self, prompt, limit, prepared=None):
        self.calls += 1
        time.sleep(.01)
        return {'answer': f'STUB ONLY: {self.loaded_id}', 'output_tokens': 1}


class TransformersBackend:
    """Inference using the exact runtime: evaluation's chat template and decoding."""
    def __init__(self, manifest, artifact):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        runtime = manifest['runtime']
        generation = runtime['generation']
        revision = runtime['base_revision']
        if len(revision) != 40 or generation['do_sample'] is not False:
            raise ValueError('Нужны закреплённая ревизия и детерминированная генерация')
        if runtime['dtype'] not in ('bfloat16', 'float16', 'float32'):
            raise ValueError('Неизвестный dtype')
        if not isinstance(generation.get('system'), str):
            raise ValueError('Отсутствует system-инструкция оценки')
        self.torch = torch
        self.runtime = runtime
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        torch.manual_seed(int(generation.get('seed', 42)))
        kind = manifest.get('model_kind', 'base')
        if kind == 'base':
            model_path = str(artifact)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=getattr(torch, runtime['dtype']), local_files_only=True)
        elif kind == 'adapter':
            from peft import PeftModel
            base_name = runtime['base_model']
            self.tokenizer = AutoTokenizer.from_pretrained(str(artifact), local_files_only=True)
            base = AutoModelForCausalLM.from_pretrained(base_name, revision=revision, dtype=getattr(torch, runtime['dtype']), local_files_only=True)
            self.model = PeftModel.from_pretrained(base, str(artifact), local_files_only=True)
        else:
            raise ValueError('Неизвестный вид модели')
        self.model = self.model.to(self.device).eval()

    def warmup(self):
        prompt = 'Title: Startup test\nBody: The application fails to open the file.\nChoices: bug, documentation, feature, question'
        self.generate(prompt, self.runtime['generation']['max_new_tokens'], self.validate_prompt(prompt))

    def validate_prompt(self, prompt):
        generation = self.runtime['generation']
        text = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': generation['system']}, {'role': 'user', 'content': prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=generation['enable_thinking'])
        batch = self.tokenizer(text, return_tensors='pt').to(self.device)
        if batch['input_ids'].shape[1] > int(self.runtime.get('max_input_tokens', 4096)):
            raise ValueError('Токенизированный запрос слишком длинный')
        return batch

    def generate(self, prompt, limit, prepared=None):
        batch = prepared if prepared is not None else self.validate_prompt(prompt)
        self.calls += 1
        with self.torch.inference_mode():
            result = self.model.generate(**batch, max_new_tokens=limit, do_sample=False,
                                         pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        generated = result[0, batch['input_ids'].shape[1]:]
        return {'answer': self.tokenizer.decode(generated, skip_special_tokens=True), 'output_tokens': int(generated.shape[0])}


class ServiceState:
    def __init__(self, root, warmup_seconds=0, backend_kind='stub', manifest_path=None, backend_factory=None):
        self.root = Path(root)
        self.warmup_seconds = warmup_seconds
        self.backend_kind = backend_kind
        self.manifest_path = Path(manifest_path) if manifest_path else self.root / 'accepted.json'
        self.diagnostic_only = manifest_path is not None
        self.backend_factory = backend_factory
        self.backend = None
        self.loaded_id = None
        self.loaded_runtime = None
        self.error = None
        self.warmup_finished = threading.Event()
        self.lock = threading.Lock()
        self.limit = 32

    def initialize(self):
        try:
            manifest = json.loads(self.manifest_path.read_text())
            artifact = check_manifest(manifest, self.root)
            self.limit = manifest['runtime']['generation']['max_new_tokens']
            if type(self.limit) is not int or not 1 <= self.limit <= 4096:
                raise ValueError('Некорректный лимит генерации в manifest')
            factory = self.backend_factory or (StubBackend if self.backend_kind == 'stub' else TransformersBackend)
            backend = factory(manifest, artifact)
            self.backend = backend
            self.loaded_id = backend.loaded_id
            self.loaded_runtime = backend.runtime_sha256
            backend.warmup()
            if self.warmup_seconds:
                time.sleep(self.warmup_seconds)
            self.warmup_finished.set()
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            print('LOAD ERROR:', self.error, flush=True)

    def generate(self, payload):
        if not self.warmup_finished.is_set():
            raise HTTPException(status_code=503, detail='Модель ещё не готова')
        if not isinstance(payload, dict) or set(payload) - {'prompt', 'max_new_tokens'}:
            raise HTTPException(status_code=400, detail='Некорректный запрос')
        prompt, limit = payload.get('prompt'), payload.get('max_new_tokens', self.limit)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8192:
            raise HTTPException(status_code=400, detail='prompt должен быть непустой строкой длиной не более 8192 символов')
        if type(limit) is not int or not 1 <= limit <= self.limit:
            raise HTTPException(status_code=400, detail='Недопустимый max_new_tokens')
        if self.backend_kind != 'stub' and limit != self.limit:
            raise HTTPException(status_code=400, detail='Для этой модели закреплён лимит оценки')
        started = time.perf_counter()
        try:
            prepared = self.backend.validate_prompt(prompt)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        with self.lock:
            begin_generation = time.perf_counter()
            try:
                result = self.backend.generate(prompt, limit, prepared)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            finished = time.perf_counter()
        result.update(model_id=self.loaded_id, runtime_sha256=self.loaded_runtime,
                      request_id=str(uuid.uuid4()), generation_seconds=finished-begin_generation,
                      waiting_seconds=begin_generation-started, total_seconds=finished-started,
                      diagnostic_only=self.diagnostic_only)
        return result


def create_app(root, warmup_seconds=0, backend_kind='stub', manifest_path=None, backend_factory=None):
    state = ServiceState(root, warmup_seconds, backend_kind, manifest_path, backend_factory)

    @asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(target=state.initialize, daemon=True)
        thread.start()
        yield

    app = FastAPI(title='HW 8–10 inference', lifespan=lifespan)
    app.state.service = state

    @app.get('/health')
    def health():
        return {'alive': True, 'backend': 'STUB ONLY' if backend_kind == 'stub' else 'transformers'}

    @app.get('/ready')
    def ready():
        if not state.warmup_finished.is_set():
            raise HTTPException(status_code=503, detail='Модель ещё не готова')
        return {'ready': True, 'model_id': state.loaded_id, 'runtime_sha256': state.loaded_runtime}

    if backend_kind == 'stub':
        @app.get('/debug')
        def debug():
            return {'loaded_id': state.loaded_id, 'calls': state.backend.calls if state.backend else 0,
                    'warmup_finished': state.warmup_finished.is_set(), 'load_error': state.error}

    @app.post('/generate')
    async def generate(request: Request):
        if int(request.headers.get('content-length', '0')) > 65536:
            raise HTTPException(status_code=413, detail='Слишком большой запрос')
        body = await request.body()
        if len(body) > 65536:
            raise HTTPException(status_code=413, detail='Слишком большой запрос')
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail='Ожидался JSON')
        return await anyio.to_thread.run_sync(state.generate, payload)

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.lab'))
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--warmup-seconds', type=float, default=0)
    parser.add_argument('--backend', choices=('stub', 'transformers'), default='stub')
    parser.add_argument('--diagnostic-manifest', type=Path, help='Явная диагностическая загрузка без публикации')
    args = parser.parse_args()
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error('--warmup-seconds должен быть неотрицательным')
    app = create_app(args.workspace, args.warmup_seconds, args.backend, args.diagnostic_manifest)
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
