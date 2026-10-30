"""Воспроизвести Prefect-сценарии на копии настоящего golden set и STUB-выпуске."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import release


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--hw67',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--prefect-home',type=Path,required=True)
    p.add_argument('--logs',type=Path,default=Path('evidence'))
    args=p.parse_args()
    hw67=args.hw67.resolve(); root=args.workspace.resolve(); logs=args.logs.resolve()
    if root.exists():
        raise FileExistsError('Для опыта нужен новый --workspace')
    logs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PREFECT_HOME']=str(args.prefect_home.resolve());env['PREFECT_SERVER_ANALYTICS_ENABLED']='false'
    cwd=Path(__file__).resolve().parent
    commands=[]
    def run(name,arguments,expected):
        command=[sys.executable,*arguments]
        completed=subprocess.run(command,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (logs/f'prefect-{name}.log').write_text('$ '+ ' '.join(command)+'\n'+completed.stdout+f'\nexit_code={completed.returncode}\n',encoding='utf-8')
        print(name, 'exit_code=',completed.returncode,flush=True)
        if (completed.returncode==0) != expected:
            raise RuntimeError(f'Неожиданный код возврата в {name}')
        commands.append({'step':name,'exit_code':completed.returncode})
    run('00-init',['lab.py','--workspace',str(root),'init'],True)
    source=hw67/'data'/'eval'/'golden.jsonl'
    shutil.copyfile(source,root/'input.txt')
    params=json.loads((hw67/'evaluation'/'params.json').read_text())
    baseline=json.loads((hw67/'evaluation'/'baseline.json').read_text())
    config={'input_format':'golden-jsonl','tokenizer_revision':baseline['revision'],
            'template_sha256':release.sha256(release.canonical({'system':params['system'],
                'enable_thinking':params['enable_thinking']})),
            'max_length':8192,'code_revision':release.sha256((cwd/'lab.py').read_bytes())}
    (root/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    flow=['prefect_flow.py','--workspace',str(root)]
    run('01-fail-after-prepare',flow+['--fail-after-prepare'],False)
    run('02-recovery',flow,True)
    content=(root/'input.txt').read_text();(root/'input.txt').write_text(' '+content)
    run('03-changed-input',flow,True)
    a=flow+['--manifest','manifest-A.json','--report','report-A.json']
    run('04-fail-after-pointer',a+['--fail-after-publish'],False)
    run('05-recover-pointer',a,True)
    run('06-repeat',a,True)
    run('07-switch-back',flow,True)
    events=[json.loads(line) for line in (root/'publications.jsonl').read_text().splitlines()]
    if [e['model_id'] for e in events] != ['fixture-B','fixture-A','fixture-B']:
        raise AssertionError('Нарушен журнал событий B → A → B')
    (logs/'prefect-summary.json').write_text(json.dumps({'steps':commands,'publication_models':[e['model_id'] for e in events],
        'golden_sha256':release.sha256(source.read_bytes()),'rows':len(source.read_text().splitlines()),
        'max_length_chars':8192,'max_input_length_chars':max(len(json.loads(line)['input']) for line in source.read_text().splitlines()),
        'note':'Проверка схемы реального golden; публикации исключительно STUB.'},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
