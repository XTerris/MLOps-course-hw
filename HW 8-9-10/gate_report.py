"""Связать прежние отчёты с решением quality_gate, сохраняя отказы."""
import argparse
import json
import os
import sys
from pathlib import Path
import release


def make_report(hw67: Path, manifest: dict) -> dict:
    hw67 = hw67.resolve()
    sys.path.insert(0, str(hw67))
    from evaluation.cli import inputs
    from evaluation.contract import fingerprint, read_json
    from evaluation.solution import quality_gate, score

    paths = {name: hw67/'metrics'/f'{name}.json' for name in
             ('baseline-eval', 'candidate-eval', 'baseline-outputs', 'candidate-outputs')}
    old_cwd = Path.cwd()
    try:
        os.chdir(hw67)
        config, rows, behavioral, baseline_spec, _, protocol_hash = inputs()
    finally:
        os.chdir(old_cwd)
    baseline = read_json(paths['baseline-eval'])
    candidate = read_json(paths['candidate-eval'])
    candidate_spec = read_json(hw67/'metrics'/'candidate-model.json')
    if baseline.get('model') != baseline_spec or candidate.get('model') != candidate_spec:
        raise ValueError('Отчёт относится к другой модели')
    for role, report in [('baseline', baseline), ('candidate', candidate)]:
        if (report.get('protocol_sha256') != protocol_hash or
                report.get('golden_sha256') != fingerprint(rows) or
                report.get('baseline_spec_sha256') != fingerprint(baseline_spec)):
            raise ValueError('Данные или протокол оценки изменились')
        outputs = read_json(paths[role+'-outputs'])
        if fingerprint(outputs) != report.get('outputs_sha256'):
            raise ValueError('Сырые ответы изменились')
        by_id = {item['id']: item for item in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {item['id'] for item in rows + behavioral}:
            raise ValueError('ID сырых ответов изменились')
        recalculated = score(rows, [by_id[row['id']] for row in rows])
        if any(report.get(k) != recalculated[k] for k in ('overall','slices','per_item')):
            raise ValueError('Метрики не соответствуют ответам')
        if report.get('behavioral') != score(behavioral, [by_id[row['id']] for row in behavioral]):
            raise ValueError('Behavioral-отчёт не соответствует ответам')
    if manifest.get('model_id') != candidate_spec.get('uri'):
        raise ValueError('Manifest относится к другому кандидату')
    if manifest.get('protocol_sha256') != protocol_hash or manifest.get('golden_sha256') != fingerprint(rows):
        raise ValueError('Manifest относится к другому протоколу или golden set')
    sources = {name: {'path': f'HW 6-7/metrics/{name}.json', 'sha256': release.sha256(path.read_bytes())}
               for name, path in paths.items()}
    result = {key:manifest.get(key) for key in ('model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256')}
    result.update(status='complete', passed=False,
                  evaluation_run_id=candidate.get('evaluation_run_id',''),
                  metrics={key:candidate['overall'][key] for key in ('accuracy','format_rate')},
                  source_reports=sources)
    try:
        decision = quality_gate(baseline, candidate, config['critical_slices'])
        if not isinstance(decision,dict) or decision.get('passed') is not True:
            raise ValueError('Проверка не вернула PASS')
        # Старые отчёты не связывают evaluation с хешем файлов и runtime.
        # Положительный результат допустим только при наличии такой связи в самом отчёте.
        if any(candidate.get(k) != manifest.get(k) for k in ('model_sha256','runtime_sha256')):
            raise ValueError('Отчёт оценки не фиксирует хеши model и runtime; нужна новая оценка')
        result.update(passed=True, gate=decision)
    except ValueError as exc:
        result['reason'] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw67', type=Path, default=Path(__file__).resolve().parent.parent/'HW 6-7')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    try:
        report = make_report(args.hw67, manifest)
    except Exception as exc:
        report = {'status':'incomplete','passed':False,'reason':str(exc)}
    release.atomic_write(args.output, (json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode())
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report.get('passed') is True else 2


if __name__ == '__main__':
    raise SystemExit(main())
