"""Create a diagnostic manifest for the old baseline; this is not a release."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import yaml
from service import canonical, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw67', type=Path, default=Path('../HW 6-7'))
    parser.add_argument('--output', type=Path, default=Path('results/baseline-diagnostic-manifest.json'))
    args = parser.parse_args()
    hw67 = args.hw67.resolve()
    config = json.loads((hw67 / 'evaluation/params.json').read_text())
    baseline = json.loads((hw67 / 'evaluation/baseline.json').read_text())
    evaluation = json.loads((hw67 / 'metrics/baseline-eval.json').read_text())
    training = yaml.safe_load((hw67 / 'params.yaml').read_text())
    if baseline != evaluation['model'] or baseline['kind'] != 'base':
        raise ValueError('Базовая модель и прошлый eval не совпадают')
    revision = baseline['revision']
    snapshot = Path(snapshot_download(repo_id=baseline['name'], revision=revision, local_files_only=True))
    if snapshot.name != revision:
        raise ValueError('Ревизия снимка не совпадает с оценкой')
    names = ['config.json', 'generation_config.json', 'model.safetensors', 'tokenizer.json',
             'tokenizer_config.json', 'vocab.json', 'merges.txt']
    file_hashes = {name: sha256(snapshot / name) for name in names}
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer_files = {name: file_hashes[name] for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}
    runtime = {
        'base_model': baseline['name'],
        'base_revision': revision,
        'tokenizer_sha256': hashlib.sha256(canonical(tokenizer_files)).hexdigest(),
        'template_sha256': hashlib.sha256(canonical(tokenizer.chat_template)).hexdigest(),
        'dtype': training['model']['dtype'],
        'max_input_tokens': 4096,
        'generation': {'system': config['system'], 'do_sample': False, 'seed': config['seed'],
                       'max_new_tokens': config['max_new_tokens'], 'enable_thinking': config['enable_thinking']},
    }
    manifest = {
        'schema_version': 1, 'model_id': baseline['name'] + '@' + revision,
        'model_kind': 'base', 'artifact_uri': str(snapshot), 'file_hashes': file_hashes,
        'model_sha256': hashlib.sha256(canonical(file_hashes)).hexdigest(),
        'protocol_sha256': evaluation['protocol_sha256'], 'golden_sha256': evaluation['golden_sha256'],
        'runtime': runtime, 'runtime_sha256': hashlib.sha256(canonical(runtime)).hexdigest(),
        'origin': 'baseline diagnostic only; not an accepted release',
        'baseline_evaluation_run_id': evaluation['evaluation_run_id'],
        'baseline_evaluation_sha256': sha256(hw67 / 'metrics/baseline-eval.json'),
        'baseline_outputs_sha256': sha256(hw67 / 'metrics/baseline-outputs.json'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print('DIAGNOSTIC ONLY', args.output, manifest['model_id'], manifest['runtime_sha256'])


if __name__ == '__main__':
    main()
