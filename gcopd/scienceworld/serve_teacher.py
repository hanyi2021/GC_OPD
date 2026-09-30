"""Start an explicitly configured frozen source-generation service."""
import argparse
import subprocess
import sys
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',required=True);p.add_argument('--served-model',default='teacher')
p.add_argument('--tensor-parallel-size',type=int,default=4);p.add_argument('--port',type=int,default=8000)
a=p.parse_args()
raise SystemExit(subprocess.call([sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',a.model,'--served-model-name',a.served_model,'--tensor-parallel-size',str(a.tensor_parallel_size),'--port',str(a.port),'--max-model-len','10753','--dtype','bfloat16','--enable-prefix-caching']))
