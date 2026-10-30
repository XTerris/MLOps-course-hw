"""Запустить реальный кандидат и проверить неизменность принятого указателя."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import release


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--prefect-home',type=Path,required=True)
    parser.add_argument('--log',type=Path,default=Path('evidence/prefect-real-candidate-refusal.log'))
    args=parser.parse_args()
    root=args.workspace.resolve();target=root/'accepted.json';log=args.log.resolve()
    before=release.sha256(target.read_bytes())
    env=os.environ.copy();env['PREFECT_HOME']=str(args.prefect_home.resolve());env['PREFECT_SERVER_ANALYTICS_ENABLED']='false'
    command=[sys.executable,str(Path(__file__).resolve().parent/'prefect_flow.py'),
             '--workspace',str(root),'--manifest',str(args.manifest.resolve()),'--report',str(args.report.resolve())]
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env)
    after=release.sha256(target.read_bytes())
    log.parent.mkdir(parents=True,exist_ok=True)
    log.write_text('$ '+' '.join(command)+'\n'+result.stdout+f'\nexit_code={result.returncode}\naccepted_before_sha256={before}\naccepted_after_sha256={after}\n',encoding='utf-8')
    print('exit_code=',result.returncode,'accepted_before=',before,'accepted_after=',after)
    if result.returncode==0 or before!=after or 'GATE_REFUSE' not in result.stdout:
        raise RuntimeError('Проверка честного отказа не пройдена')


if __name__=='__main__':
    main()
