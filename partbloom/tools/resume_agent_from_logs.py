"""Rebuild a failed Agent canvas from identical cached calls, then continue API calls.

Use a fresh output folder. No successful model response is regenerated or edited.
The complete rebuilt request must match the saved request before a response is replayed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys

PACKAGE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PACKAGE),str(PACKAGE/'pipeline')]
import api_client


def replay_client(source_logs):
    class ReplayClient(api_client.Client):
        def call(self,inputs,label,tools=None):
            previous=source_logs/f'{label}_response.json'
            if not previous.exists():return super().call(inputs,label,tools)
            payload=self.adapter.payload(inputs,tools,self.model,self.reasoning_effort)
            saved=json.loads((source_logs/f'{label}_request.json').read_text())
            if payload!=saved:raise ValueError('Cached request differs; cannot replay '+label)
            response=json.loads(previous.read_text())
            if response.get('model')!=self.model:raise ValueError('Cached model differs')
            for suffix in ['request','response']:
                shutil.copy2(source_logs/f'{label}_{suffix}.json',self.logs/f'{label}_{suffix}.json')
            self.calls.append({'label':label,'replayed':True,'source':str(previous),
                'source_sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),
                'seconds':0,'attempts':0,'usage':response.get('usage'),'model':self.model,
                'reasoning_effort':self.reasoning_effort,'protocol':self.protocol})
            (self.logs/'calls.json').write_text(json.dumps(self.calls,indent=2))
            print('API_REPLAY',label,flush=True)
            return response
    return ReplayClient


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-logs',required=True,type=Path)
    args,agent_args=parser.parse_known_args()
    api_client.Client=replay_client(args.source_logs.resolve())
    sys.argv=['agent_runner.py',*agent_args]
    runpy.run_path(str(PACKAGE/'pipeline/agent_runner.py'),run_name='__main__')


if __name__=='__main__':main()
