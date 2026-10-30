"""Исполняемый прототип для расследования. Только синтетические данные, без модели.

Запускайте в отдельном каталоге .lab. Этот последовательный прототип надо
превратить в корректный Prefect-flow; его зелёный статус не доказывает корректность.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path
import release


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def initialize(root):
    if root.exists():
        raise FileExistsError('Рабочий каталог уже существует. Выберите новый --workspace.')
    root.mkdir(parents=True)
    runtime={'base_revision':'fixture-only','tokenizer_sha256':'a'*64,
             'template_sha256':'b'*64,'generation':{'max_new_tokens':32,'do_sample':False},'dtype':'float32'}
    for version in ['A','B']:
        artifact=root/f'model-{version}.txt';artifact.write_text(f'STUB ONLY — version {version}\n')
        files={artifact.name:digest(artifact.read_bytes())}
        m={'schema_version':1,'model_id':f'fixture-{version}',
           'artifact_uri':artifact.name,'model_sha256':digest(canonical(files)),
           'protocol_sha256':'d'*64,'golden_sha256':'e'*64,
           'runtime_sha256':digest(canonical(runtime)),'runtime':runtime,'file_hashes':files}
        report={k:m[k] for k in ['model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256']}
        report.update(status='complete',passed=True,evaluation_run_id=f'fixture-eval-{version}',metrics={'accuracy':.8,'format_rate':1.0},synthetic=True)
        write(root/f'manifest-{version}.json',m);write(root/f'report-{version}.json',report)
    write(root/'accepted.json',read(root/'manifest-A.json'))
    report=read(root/'report-B.json');report['status']='running';report['passed']='false'
    write(root/'report-incomplete.json',report)
    report=read(root/'report-B.json');report['passed']=False
    write(root/'report-reject.json',report)
    write(root/'config.json',{'tokenizer_revision':'r1','template_sha256':'b'*64,'max_length':128,'code_revision':'prepare-v1'})
    (root/'input.txt').write_text('Первый пример\nВторой пример\n')
    print('Создан синтетический стенд:',root)


def prepare(root):
    config=read(root/'config.json');key=release.cache_key(root/'input.txt',config)
    cache=root/'cache';cache.mkdir(exist_ok=True);entry=cache/(key+'.json')
    if entry.exists():
        result=read(entry);print('CACHE_HIT',result['input_sha256']);return result
    payload=(root/'input.txt').read_bytes()
    lines=payload.decode('utf-8').splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError('Пустой набор или пустая строка')
    result={'input_sha256':digest(payload),'rows':len(lines),
            'word_counts':[len(line.split()) for line in lines], 'config':config}
    if config.get('input_format') == 'golden-jsonl':
        max_length = config.get('max_length')
        if type(max_length) is not int or max_length < 1:
            raise ValueError('max_length должен задавать положительное число символов')
        rows = [json.loads(line) for line in lines]
        ids = set()
        for row in rows:
            if (not isinstance(row,dict) or not isinstance(row.get('id'),str) or not row['id']
                or row['id'] in ids or not isinstance(row.get('input'),str) or not row['input'].strip()
                or len(row['input']) > max_length
                or not isinstance(row.get('choices'),list) or row.get('expected') not in row['choices']
                or not isinstance(row.get('slices'),list) or not row['slices']
                or not isinstance(row.get('source'),str) or not row['source']):
                raise ValueError('Некорректная схема golden JSONL')
            ids.add(row['id'])
        result['schema']='golden-jsonl-v1'
        result['max_input_length_chars']=max(len(row['input']) for row in rows)
    release.atomic_write(entry,canonical(result))
    print('PREPARED',result['input_sha256']);return result


def record_publication(root, event):
    """События и указатель согласуются при повторе после сбоя одного писателя."""
    path=root/'publications.jsonl'
    existing=path.read_text(encoding='utf-8') if path.exists() else ''
    events=[json.loads(line) for line in existing.splitlines() if line.strip()]
    if not any(item['event_id']==event['event_id'] for item in events):
        release.atomic_write(path,(existing + json.dumps(event,ensure_ascii=False)+'\n').encode('utf-8'))
        print('PUBLICATION_EVENT',event['release_id'])
    else:
        print('PUBLICATION_ALREADY_RECORDED',event['release_id'])


def publish_stage(root,manifest,report,prepared,fail_after_publish=False):
    release.validate_release(manifest, report)
    release.verify_files(manifest, root)
    release_id=digest(canonical({'model_sha256':manifest['model_sha256'],
                                  'model_id':manifest['model_id'],
                                  'runtime_sha256':manifest['runtime_sha256'],
                                  'report_sha256':digest(canonical(report))}))
    pending_path=root/'pending-publication.json'
    target=root/'accepted.json'
    current=read(target) if target.exists() else None
    log_path=root/'publications.jsonl'
    prior=[json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines() if line.strip()] if log_path.exists() else []
    if pending_path.exists():
        event=read(pending_path)
        if event['release_id'] != release_id:
            raise ValueError('Сначала завершите предыдущую прерванную публикацию')
    elif current == manifest and prior and prior[-1]['release_id'] == release_id:
        print('PUBLICATION_ALREADY_CURRENT',release_id)
        return
    else:
        event={'event_id':str(uuid.uuid4()),'release_id':release_id,
               'model_id':manifest['model_id'],'runtime_sha256':manifest['runtime_sha256'],
               'prepared':prepared['input_sha256']}
        release.atomic_write(pending_path,canonical(event))
    release.publish(manifest,report,root/'accepted.json')
    if fail_after_publish:raise RuntimeError('Инъекция сбоя после указателя до журнала')
    record_publication(root,event)
    pending_path.unlink()
    print('PUBLISHED',manifest['model_id'])


def run(root,report_name,fail_after_prepare=False,manifest_name='manifest-B.json',fail_after_publish=False):
    attempt={'attempt_id':str(uuid.uuid4()),'started_at':time.time()}
    try:
        _run(root,report_name,fail_after_prepare,manifest_name,fail_after_publish)
        attempt['status']='success'
    except Exception as exc:
        attempt.update(status='failed',error=str(exc))
        raise
    finally:
        with (root/'attempts.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(attempt,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())


def _run(root,report_name,fail_after_prepare,manifest_name,fail_after_publish):
    prepared=prepare(root)
    if fail_after_prepare:raise RuntimeError('Инъекция сбоя после подготовки')
    manifest=read(root/manifest_name);report=read(root/report_name)
    release.validate_stub(manifest, report, root)
    publish_stage(root,manifest,report,prepared,fail_after_publish)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path('.lab'))
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('init')
    run_parser=sub.add_parser('run');run_parser.add_argument('--report',default='report-B.json');run_parser.add_argument('--manifest',default='manifest-B.json');run_parser.add_argument('--fail-after-prepare',action='store_true');run_parser.add_argument('--fail-after-publish',action='store_true')
    args=parser.parse_args()
    if args.command=='init':initialize(args.workspace)
    else:run(args.workspace,args.report,args.fail_after_prepare,args.manifest,args.fail_after_publish)


if __name__=='__main__':main()
