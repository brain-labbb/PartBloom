"""Archive a failed API attempt, replay its successful responses, finish the same run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PACKAGE=Path(__file__).resolve().parents[1]
ROOT=PACKAGE.parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--folder',required=True,type=Path)
    p.add_argument('--api-config',required=True,type=Path)
    p.add_argument('--gpu',required=True)
    a=p.parse_args();folder=a.folder.resolve();coarse=folder/'coarse'
    state=json.loads((folder/'experiment_status.json').read_text())
    agent=json.loads((coarse/'agent_status.json').read_text())
    if agent['status']!='failed':raise ValueError('Only recover a failed Agent stage')
    attempt=1
    while (folder/f'coarse_failed_attempt{attempt}').exists():attempt+=1
    archive=folder/f'coarse_failed_attempt{attempt}'
    coarse.rename(archive)
    (folder/'coarse.log').rename(folder/f'coarse_failed_attempt{attempt}.log')
    state.setdefault('recovery_attempts',[]).append({'source':str(archive),'original_seconds':agent.get('seconds'),
        'reason':'API failure; exact successful request/response replay','gpu':a.gpu,'started':time.time()})
    state.pop('error',None)
    def save(status):
        state['status']=status
        (folder/'experiment_status.json').write_text(json.dumps(state,indent=2))
        print(folder.name,status,flush=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    def call(log,script,args):
        with (folder/log).open('w') as output:
            subprocess.run([sys.executable,str(script),*map(str,args)],stdout=output,stderr=subprocess.STDOUT,
                           env=env,check=True)
    try:
        save('coarse_recovering')
        call('coarse.log',PACKAGE/'tools/resume_agent_from_logs.py',[
            '--source-logs',archive/'api_logs','--image',folder/'input.png','--out',coarse,
            '--api-config',a.api_config.resolve(),'--model',state['model'],
            '--reasoning-effort',state['reasoning_effort'],'--repairs',2,'--max-api-steps',10])
        save('refine_running')
        call('refine.log',PACKAGE/'run.py',['refine','--source',coarse,'--out',folder/'refined',
                                         '--gpu',a.gpu,'--decode-mode','parts'])
        save('render_running')
        call('render.log',PACKAGE/'run.py',['render','--source',folder/'refined/objs','--out',folder/'parts.png'])
        state['seconds']=time.time()-state['started']
        save('complete')
    except Exception as exc:
        state['error']=str(exc);save('failed');raise


if __name__=='__main__':main()
