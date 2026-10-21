"""Дополнительный отрицательный тест с согласованными counts и per_item."""
from pathlib import Path
import copy,json,hashlib
from evaluation.contract import read_json,read_rows,write_json
from evaluation.solution import score,quality_gate
from src.common import connect,load_params
from mlflow import MlflowClient

def main():
 params=read_json('evaluation/params.json');rows=read_rows(params['golden'])
 baseline=read_json('metrics/baseline-eval.json');broken=read_json('metrics/candidate-eval.json')
 path=Path('metrics/candidate-outputs.json');before=hashlib.sha256(path.read_bytes()).hexdigest()
 outputs=copy.deepcopy(read_json(path));ids={r['id'] for r in rows}
 for output in outputs:
  if output['id'] in ids:output.update(raw='INVALID',error=None)
 by_id={o['id']:o for o in outputs};broken.update(score(rows,[by_id[r['id']] for r in rows]))
 connect(load_params());client=MlflowClient();name=load_params()['registry']['name']
 aliases_before=client.get_registered_model(name).aliases
 try:quality_gate(baseline,broken,params['critical_slices'])
 except ValueError as exc:reason=str(exc)
 else:raise AssertionError('Corrupted answers passed')
 assert reason=='Регрессия общей accuracy',reason
 assert before==hashlib.sha256(path.read_bytes()).hexdigest()
 assert aliases_before==client.get_registered_model(name).aliases
 write_json('metrics/negative-outputs.json',outputs)
 write_json('metrics/negative-raw-proof.json',dict(rejected=True,reason=reason,overall=broken['overall'],
  source_outputs_sha256=before,original_outputs_unchanged=True,aliases_unchanged=True,
  note='Only copied golden raw answers changed; behavioral metrics and all counts remain internally consistent.'))
 print('PASS: consistently scored corrupted raw answers rejected; original outputs and aliases unchanged')
if __name__=='__main__':main()
