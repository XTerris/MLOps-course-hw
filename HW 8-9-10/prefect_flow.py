"""Локальный Prefect-flow. Передаются пути и метаданные, не веса модели."""
import argparse
import json
import os
import time
from pathlib import Path

# Настраивается до импорта Prefect: и база задач, и результаты сохраняются локально.
os.environ.setdefault('PREFECT_HOME', str(Path(__file__).resolve().parent/'.prefect'))
os.environ.setdefault('PREFECT_LOCAL_STORAGE_PATH', str(Path(os.environ['PREFECT_HOME'])/'storage'))
os.environ.setdefault('PREFECT_SERVER_ANALYTICS_ENABLED', 'false')

from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.context import get_run_context
import lab
import release


def preparation_key(_context, parameters):
    root = Path(parameters['workspace'])
    return release.cache_key(root/'input.txt', lab.read(root/'config.json'))


@task(name='prepare-input', cache_key_fn=preparation_key, persist_result=True, log_prints=True)
def prepare_input(workspace: str):
    return lab.prepare(Path(workspace))


@task(name='obtain-and-verify-artifact', cache_policy=NO_CACHE, log_prints=True)
def obtain_artifact(workspace: str, manifest_name: str):
    root = Path(workspace)
    manifest = lab.read(root/manifest_name)
    release.verify_files(manifest, root)
    print('ARTIFACT_VERIFIED', manifest['model_id'])
    return manifest


@task(name='evaluate-release', cache_policy=NO_CACHE, log_prints=True)
def evaluate_release(workspace: str, report_name: str, manifest: dict):
    root = Path(workspace)
    report = lab.read(root/report_name)
    if report.get('synthetic') is True:
        release.validate_stub(manifest, report, root)
    else:
        import gate_report
        current = gate_report.make_report(Path(__file__).resolve().parent.parent/'HW 6-7', manifest)
        # Проверка выполняется заново, а не только по переданному passed.
        if current != report:
            raise ValueError('Отчёт gate не соответствует текущим исходным данным')
    if report.get('passed') is not True:
        print('GATE_REFUSE',report.get('reason','неполная оценка'))
    release.validate_release(manifest, report)
    print('GATE_PASS',manifest['model_id'])
    return report


@task(name='publish-release', cache_policy=NO_CACHE, log_prints=True)
def publish_release(workspace: str, manifest: dict, report: dict, prepared: dict, fail_after_publish: bool):
    lab.publish_stage(Path(workspace), manifest, report, prepared, fail_after_publish)


@flow(name='model-release', log_prints=True)
def release_flow(workspace: str, manifest_name: str='manifest-B.json', report_name: str='report-B.json',
                 fail_after_prepare: bool=False, fail_after_publish: bool=False):
    attempt={'attempt_id':str(get_run_context().flow_run.id),'started_at':time.time()}
    print('FLOW_RUN_ID',attempt['attempt_id'])
    try:
        state = prepare_input(workspace, return_state=True)
        print('PREPARE_TASK_STATE',state.name)
        prepared = state.result()
        if fail_after_prepare:
            raise RuntimeError('Инъекция сбоя после подготовки')
        manifest = obtain_artifact(workspace, manifest_name)
        report = evaluate_release(workspace, report_name, manifest)
        publish_release(workspace, manifest, report, prepared, fail_after_publish)
        attempt['status']='success'
    except Exception as exc:
        attempt.update(status='failed',error=str(exc))
        raise
    finally:
        with (Path(workspace)/'attempts.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(attempt,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--manifest', default='manifest-B.json')
    parser.add_argument('--report', default='report-B.json')
    parser.add_argument('--fail-after-prepare', action='store_true')
    parser.add_argument('--fail-after-publish', action='store_true')
    args = parser.parse_args()
    release_flow(str(args.workspace.resolve()), args.manifest, args.report,
                 args.fail_after_prepare, args.fail_after_publish)


if __name__ == '__main__':
    main()
