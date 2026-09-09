import sys,os,json,gc
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from src.config import load_params
from src.model import load_model
from src.inspect_model import activation_norms,PeakMemory,device_allocated_bytes
p=load_params()
def hook_checks():
 tokenizer,model=load_model(p)
 model.train();model.get_decoder().layers[1].eval()
 flags=[m.training for m in model.modules()]
 count=lambda:sum(len(m._forward_hooks) for m in model.modules())
 before=count()
 a=activation_norms(tokenizer,model,p);first=count()
 b=activation_norms(tokenizer,model,p);second=count()
 diff=max(abs(x-y) for k in a['norms'] for x,y in zip(a['norms'][k],b['norms'][k]))
 result={'hooks_before':before,'hooks_after_first':first,'hooks_after_second':second,'max_abs_norm_difference':diff,'training_flags_restored':[m.training for m in model.modules()]==flags}
 assert (before,first,second,diff,result['training_flags_restored'])==(0,0,0,0,True)
 return result
result={'hooks':hook_checks()}
gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
with PeakMemory(torch.device('mps')) as peak:
 temporary=torch.ones(16*1024*1024,device='mps',dtype=torch.float32)
 peak.checkpoint('temporary_64_mib')
 del temporary
 gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
 peak.checkpoint('released')
result['transient_allocation']=peak.result()
result['transient_allocation']['last_bytes']=device_allocated_bytes(torch.device('mps'))
assert peak.used>result['transient_allocation']['last_bytes']
result['shared_python']=sys.executable
Path('docs/results/diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))
