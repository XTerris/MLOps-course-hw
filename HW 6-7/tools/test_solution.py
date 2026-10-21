"""Дополнительные проверки границ контракта; исходный tests/ не меняется."""
from pathlib import Path
import copy
import sys
import unittest
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'tests')]
from evaluation.solution import score, quality_gate
from test_eval import rows, outputs, report

class EdgeCases(unittest.TestCase):
 def test_duplicate_json_keys_rejected(self):
  for raw in ['{"answer":"A","answer":"B"}', '{"answer":"B","extra":NaN}']:
   result=score(rows(1),[dict(id='q0',raw=raw,error=None)])
   self.assertEqual(result['overall']['correct'],0)
   self.assertEqual(result['overall']['valid'],0)
 def test_duplicate_question_id_rejected(self):
  with self.assertRaises(ValueError):score([rows(1)[0]]*2,outputs(rows(2)))
 def test_error_overrides_apparently_correct_raw(self):
  result=score(rows(1),[dict(id='q0',raw='{"answer":"B"}',error='timeout')])
  self.assertEqual(result['overall'],dict(n=1,correct=0,valid=0,errors=1,accuracy=0.0,format_rate=0.0))
 def test_overlapping_slices_do_not_inflate_overall(self):
  items=rows(3);items[0]['slices']+=['shared'];items[1]['slices']+=['shared']
  result=score(items,outputs(items));self.assertEqual(result['overall']['n'],3)
  self.assertEqual(result['slices']['shared']['n'],2)
 def test_forged_slice_and_behavior_counts_rejected(self):
  for section,key in [('slices','s0'),('behavioral','overall')]:
   candidate=report();candidate[section][key]['valid']-=1
   with self.assertRaises(ValueError):quality_gate(report(),candidate,['s0'])
 def test_forged_per_item_membership_rejected(self):
  candidate=report();candidate['per_item'][0]['slices']=['s1']
  with self.assertRaises(ValueError):quality_gate(report(),candidate,['s0'])
 def test_baseline_generation_errors_also_block(self):
  baseline=report();baseline['per_item'][0].update(valid=False,correct=False,error='timeout')
  baseline['overall'].update(correct=49,valid=49,errors=1,accuracy=49/50,format_rate=49/50)
  baseline['slices']['s0'].update(correct=16,valid=16,errors=1,accuracy=16/17,format_rate=16/17)
  with self.assertRaises(ValueError):quality_gate(baseline,report(),['s0'])

if __name__=='__main__':unittest.main(verbosity=2)
