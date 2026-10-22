import json
import sys
from PIL import Image
import pytest
import agent_runner as agent
from voxel_engine import VoxelCanvas
from prompts.definitions import TOOLS
import run


def edit(part='body'):
    return {'op':'brush', 'part':part, 'points':[[20,20,20]], 'radius':1}


def test_batch_limit_is_independent_of_previous_edits():
    canvas=VoxelCanvas()
    assert TOOLS[0]['parameters']['properties']['edits']['maxItems']==15
    for _ in range(17):
        result=agent.dispatch_tool(canvas,'apply_voxel_edits',{'edits':[edit()]*15})
        assert all(r['ok'] for r in result['results'])
    assert len(canvas.action_log)==255
    with pytest.raises(ValueError,match='1..15'):
        agent.dispatch_tool(canvas,'apply_voxel_edits',{'edits':[edit()]*16})
    assert len(canvas.action_log)==255


def test_coarse_forwards_independent_request_limits(tmp_path,monkeypatch):
    config=run.load_config()
    assert config['max_api_steps']==10 and config['repair_max_api_steps']==20
    calls=[]
    monkeypatch.setattr(run,'invoke',lambda script,args,*a,**kw:calls.append(args))
    run.coarse(tmp_path/'image',tmp_path/'out',tmp_path/'api',config,{})
    args=calls[0]
    assert args[args.index('--max-api-steps')+1]==10
    assert args[args.index('--repair-max-api-steps')+1]==20


def test_runner_uses_repair_budget_and_keeps_neighbor_permissions(tmp_path,monkeypatch):
    image=tmp_path/'input.png';Image.new('RGB',(4,4)).save(image)
    out=tmp_path/'out';calls=[]
    graph={'parts':[{'id':'body','parent':None},{'id':'leg','parent':'body'}]}
    class Client:
        def __init__(self,*a,**kw):self.calls=[]
        def close(self):pass
    def builder(client,model,canvas,reference,out,round_no,**kwargs):
        calls.append((round_no,kwargs))
        if kwargs.get('part_id'):
            canvas.apply(edit(kwargs['part_id']))
    monkeypatch.setattr(agent,'Client',Client)
    monkeypatch.setattr(agent,'plan_parts',lambda *a:graph)
    monkeypatch.setattr(agent,'run_builder',builder)
    monkeypatch.setattr(agent,'save_round',lambda *a:None)
    monkeypatch.setattr(agent,'render_views',lambda *a:str(image))
    monkeypatch.setattr(agent,'publish_round',lambda *a:None)
    monkeypatch.setattr(agent,'run_reviewer',lambda *a:{'issues':[{'part':'leg','problem':'contact'}]})
    monkeypatch.setattr(sys,'argv',['agent_runner.py','--image',str(image),'--out',str(out),
        '--api-config',str(tmp_path/'api.json'),'--max-api-steps','7','--repair-max-api-steps','18'])
    agent.main()
    assert [(r,k['max_api_steps']) for r,k in calls]==[(0,7),(0,7),(1,18),(2,18)]
    assert all('part_id' not in k for r,k in calls if r>0)
    assert json.loads((out/'prompt_protocol.json').read_text())['budgets']['repair_max_api_steps']==18


def test_builder_stops_early_and_repair_can_edit_related_parts(tmp_path):
    image=tmp_path/'input.png';Image.new('RGB',(4,4)).save(image)
    canvas=VoxelCanvas()
    class Client:
        requests=0
        def call(self,*a):
            self.requests+=1
            if self.requests==1:
                return {'output':[{'type':'function_call','name':'apply_voxel_edits','call_id':'x',
                                   'arguments':json.dumps({'edits':[edit('body'),edit('leg')]})}]}
            return {'output':[{'type':'message','content':[{'type':'output_text','text':'done'}]}]}
    client=Client()
    agent.run_builder(client,'test',canvas,image,tmp_path,1,max_api_steps=20,
        graph={'parts':[{'id':'body'},{'id':'leg'}]},feedback={'issues':[{'part':'leg'}]})
    assert client.requests==2
    assert canvas.parts['body'].any() and canvas.parts['leg'].any()


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_request_budget_is_rejected(tmp_path,value):
    config=tmp_path/'config.json';config.write_text(json.dumps({'repair_max_api_steps':value}))
    with pytest.raises(ValueError,match='positive integer'):run.load_config(config)
