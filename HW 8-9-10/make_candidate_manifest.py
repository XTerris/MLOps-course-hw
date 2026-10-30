"""Recover the previously evaluated MLflow adapter, without granting release approval."""
import argparse
import hashlib
import json
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from transformers import AutoTokenizer
import yaml

from service import canonical, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw67', type=Path, default=Path('../HW 6-7'))
    parser.add_argument('--tracking-uri', default='http://127.0.0.1:5006')
    parser.add_argument('--artifact-output', type=Path, default=Path('../../candidate-artifacts'))
    parser.add_argument('--output', type=Path, default=Path('results/candidate-manifest.json'))
    args = parser.parse_args()
    hw67 = args.hw67.resolve()
    candidate = json.loads((hw67 / 'metrics/candidate-model.json').read_text())
    evaluation = json.loads((hw67 / 'metrics/candidate-eval.json').read_text())
    config = json.loads((hw67 / 'evaluation/params.json').read_text())
    if evaluation['status'] != 'complete' or evaluation['model'] != candidate:
        raise ValueError('Сохранённый отчёт не соответствует кандидату')
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    version = client.get_model_version(candidate['name'], str(candidate['version']))
    if version.run_id != candidate['run_id']:
        raise ValueError('Версия модели не соответствует ранее оцененному run')
    destination = args.artifact_output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    model_root = Path(client.download_artifacts(candidate['run_id'], 'model', str(destination))).resolve()
    model_metadata = yaml.safe_load((model_root / 'MLmodel').read_text())
    adapter = (model_root / model_metadata['flavors']['python_function']['artifacts']['adapter']['path']).resolve()
    if not adapter.is_relative_to(model_root) or not adapter.is_dir():
        raise ValueError('Недопустимый путь адаптера в MLflow')
    training_manifest = json.loads((adapter / 'manifest.json').read_text())
    model = training_manifest['model']
    if model['revision'] != json.loads((hw67 / 'evaluation/baseline.json').read_text())['revision']:
        raise ValueError('Ревизия базовой модели отличается от eval')
    names = ('adapter_config.json', 'adapter_model.safetensors', 'tokenizer.json',
             'tokenizer_config.json', 'chat_template.jinja', 'manifest.json')
    files = {name: sha256(adapter / name) for name in names}
    tokenizer = AutoTokenizer.from_pretrained(str(adapter), local_files_only=True)
    tokenizer_files = {name: files[name] for name in ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    runtime = {
        'base_model': model['name'], 'base_revision': model['revision'],
        'tokenizer_sha256': hashlib.sha256(canonical(tokenizer_files)).hexdigest(),
        'template_sha256': hashlib.sha256(canonical(tokenizer.chat_template)).hexdigest(),
        'dtype': model['dtype'], 'max_input_tokens': 4096,
        'generation': {'system': config['system'], 'do_sample': False, 'seed': config['seed'],
                       'max_new_tokens': config['max_new_tokens'], 'enable_thinking': config['enable_thinking']},
    }
    manifest = {
        'schema_version': 1, 'model_id': candidate['uri'], 'model_kind': 'adapter',
        'artifact_uri': str(adapter), 'file_hashes': files,
        'model_sha256': hashlib.sha256(canonical(files)).hexdigest(),
        'protocol_sha256': evaluation['protocol_sha256'], 'golden_sha256': evaluation['golden_sha256'],
        'runtime': runtime, 'runtime_sha256': hashlib.sha256(canonical(runtime)).hexdigest(),
        'model_source_uri': candidate['uri'], 'tracking_uri': args.tracking_uri,
        'training_run_id': candidate['run_id'], 'evaluation_run_id': evaluation['evaluation_run_id'],
        'candidate_evaluation_sha256': sha256(hw67 / 'metrics/candidate-eval.json'),
        'candidate_outputs_sha256': sha256(hw67 / 'metrics/candidate-outputs.json'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('MANIFEST', args.output.resolve(), 'MODEL', manifest['model_id'],
          'MODEL_SHA256', manifest['model_sha256'], 'RUNTIME_SHA256', manifest['runtime_sha256'])
    print('ARTIFACT', adapter)


if __name__ == '__main__':
    main()
