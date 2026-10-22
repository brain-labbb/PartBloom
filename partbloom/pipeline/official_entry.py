"""Execute upstream scripts verbatim; redirect model loading to local weights only."""
import argparse,os,sys,runpy,time,json,traceback
from pathlib import Path
from resources import VENDOR, WEIGHTS_ROOT
UP=VENDOR/'PhysX-Omni'; W=WEIGHTS_ROOT
p=argparse.ArgumentParser();p.add_argument('stage',choices=['decode']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(UP))
os.environ.update(HF_HOME=str(W/'.huggingface'),TORCH_HOME=str(W/'.torch'),U2NET_HOME=str(W/'u2net'),HF_HUB_OFFLINE='1')
started=time.time();report={'stage':a.stage,'weights_root':str(W),'official_script':'decoder_each.py','changes':['local model loading only'],'seed':1}
try:
 import torch
 original_hub=torch.hub.load
 def local_hub(repo,model,*args,**kwargs):
  if repo=='facebookresearch/dinov2':
   kwargs['pretrained']=False;kwargs['source']='local'
   d=original_hub(str(VENDOR/'dinov2'),model,*args,**kwargs)
   d.load_state_dict(torch.load(W/'dinov2/dinov2_vitl14_reg4_pretrain.pth',map_location='cpu',weights_only=False),strict=True);return d
  return original_hub(repo,model,*args,**kwargs)
 torch.hub.load=local_hub
 from trellis.pipelines import TrellisImageTo3DPipeline
 original=TrellisImageTo3DPipeline.from_pretrained
 TrellisImageTo3DPipeline.from_pretrained=staticmethod(lambda path:original(str(W/'TRELLIS-image-large') if path=='microsoft/TRELLIS-image-large' else path))
 # Optional observation-only snapshots; do not change upstream mesh processing.
 if os.environ.get('PARTBLOOM_SAVE_MESH_STAGES') == '1':
  import trimesh
  from trellis.utils import postprocessing_utils as pp
  stages=a.out/'mesh_stages';stages.mkdir(exist_ok=True)
  original_post=pp.postprocess_mesh
  original_fill=pp._fill_holes
  stage_counter={'part':-1}
  def save_stage(name,verts,faces):
   folder=stages/f"part_{stage_counter['part']}";folder.mkdir(exist_ok=True)
   if torch.is_tensor(verts):verts=verts.detach().cpu().numpy()
   if torch.is_tensor(faces):faces=faces.detach().cpu().numpy()
   trimesh.Trimesh(vertices=verts,faces=faces,process=False).export(folder/(name+'.ply'))
  def observed_fill(verts,faces,*args,**kwargs):
   save_stage('simplified',verts,faces)
   result=original_fill(verts,faces,*args,**kwargs)
   save_stage('after_visibility_and_hole_fill',*result)
   return result
  def observed_post(verts,faces,*args,**kwargs):
   stage_counter['part']+=1
   save_stage('raw_decoder',verts,faces)
   result=original_post(verts,faces,*args,**kwargs)
   save_stage('postprocessed',*result)
   return result
  pp._fill_holes=observed_fill;pp.postprocess_mesh=observed_post
  report['changes'].append('Observation-only raw/simplified/postprocessed PLY snapshots')
 os.chdir(UP)
 sys.argv=['decoder_each.py' ,'--name',a.out.name,'--basepath',str(a.out.parent),'--imgpath',str(a.out/'cond_img.png')]
 runpy.run_path(str(UP/'decoder_each.py'),run_name='__main__')
 meshes=list((a.out/'objs').glob('*/*.obj'))
 if not meshes:raise RuntimeError('No official OBJ exported')
 report['obj_files']=[str(f.relative_to(a.out)) for f in meshes]
 report['status']='complete'
except Exception:
 report['status']='failed';report['traceback']=traceback.format_exc();raise
finally:
 report['seconds']=time.time()-started
 if 'torch' in sys.modules and torch.cuda.is_available():report['max_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
 (a.out/f'{a.stage}_status.json').write_text(json.dumps(report,indent=2))
