import json
import numpy as np
import pytest
from agent_runner import validate_graph,dispatch_tool,save_round
from voxel_engine import VoxelCanvas
from PIL import Image


def graph():
 return {'parts':[{'id':'seat','parent':'frame','joint_type':'revolute'},{'id':'frame','parent':None,'joint_type':'fixed'}]}


def test_geometry_plan_needs_no_physics_and_has_no_fixed_part_count():
 g={'parts':[{'id':f'part_{i}','parent':None,'bbox':[[1,1,1],[2,2,2]],'geometry':'panel'} for i in range(33)]}
 assert validate_graph(g) is g
 canvas=VoxelCanvas()
 for p in g['parts']:canvas._part(p['id'])
 assert len(canvas.parts)==33


def test_graph_and_per_part_tools():
 validate_graph(graph());bad=graph();bad['parts'][1]['parent']='seat'
 with pytest.raises(ValueError):validate_graph(bad)
 v=VoxelCanvas()
 r=dispatch_tool(v,'apply_voxel_edits',{'edits':[{'op':'brush','part':'frame','points':[[20,20,20]],'radius':2}]},allowed_parts={'seat'})
 assert r['results'][0]['ok'] is False and not v.parts


def test_repair_preserves_graph_order_and_surface_indices(tmp_path):
 v=VoxelCanvas()
 for part,x in [('frame',20),('seat',35)]:v.apply({'op':'brush','part':part,'points':[[x,20,20]],'radius':2})
 image=tmp_path/'input.png';Image.new('RGB',(8,8),'white').save(image)
 out=tmp_path/'output';save_round(v,graph(),out,image)
 assert [i['part'] for i in json.loads((out/'part_index.json').read_text())]==['seat','frame']
 v.apply({'op':'clear','part':'seat'});v.apply({'op':'brush','part':'seat','points':[[35,20,21]],'radius':2})
 save_round(v,graph(),out,image)
 assert np.array_equal(np.load(out/'allind.npy'),np.concatenate([np.load(out/'ind_0.npy'),np.load(out/'ind_1.npy')]))
 assert list(v.parts)==['seat','frame']


def test_reexport_removes_stale_indices(tmp_path):
 v=VoxelCanvas()
 for part,x in [('a',10),('b',30)]:v.apply({'op':'brush','part':part,'points':[[x,20,20]],'radius':2})
 v.export(tmp_path);v.apply({'op':'clear','part':'b'});v.export(tmp_path)
 assert not (tmp_path/'ind_1.npy').exists()
