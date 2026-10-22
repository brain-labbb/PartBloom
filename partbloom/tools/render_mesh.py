"""Blender renderer: shared fixed world cameras, no per-method rescaling."""
import argparse,sys,json,math
from pathlib import Path
import bpy
from mathutils import Vector
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--view',default='oblique',choices=['oblique','front','side','top']);p.add_argument('--part-colors',action='store_true');p.add_argument('--palette-json');p.add_argument('--explode',type=float,default=0.);p.add_argument('--ortho-scale',type=float,default=1.35);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
custom_palette=json.loads(Path(a.palette_json).read_text()) if a.palette_json else {}
bpy.ops.wm.read_factory_settings(use_empty=True)
paths=sorted(Path(a.input).glob('*/*.obj')) if Path(a.input).is_dir() else [Path(a.input)]
for index,path in enumerate(paths):
 before=set(bpy.data.objects)
 bpy.ops.wm.obj_import(filepath=str(path.resolve()),forward_axis='Y',up_axis='Z')
 for o in set(bpy.data.objects)-before:o['part_index']=index;o['part_id']=path.stem
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
if not objects:raise RuntimeError('No meshes to render')
# Neutral geometry material, same for GT/B0/B1.
for o in objects:
 palette=[(.30,.55,.90,1),(.90,.55,.25,1),(.35,.72,.40,1),(.80,.35,.55,1),(.60,.42,.85,1),(.25,.75,.75,1)]
 color=palette[int(o['part_index'])%len(palette)] if a.part_colors else (.48,.53,.60,1)
 if a.part_colors and o['part_id'] in custom_palette:color=custom_palette[o['part_id']]
 mat=bpy.data.materials.new('neutral');mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Base Color'].default_value=color;bs.inputs['Roughness'].default_value=.7;o.data.materials.clear();o.data.materials.append(mat)
 if a.explode:
  center=sum((o.matrix_world@Vector(corner) for corner in o.bound_box),Vector())/8
  o.location+=center*a.explode
loc={'oblique':(1.5,-2.5,1.5),'front':(0,-3,0),'side':(3,0,0),'top':(0,0,3)}[a.view]
bpy.ops.object.camera_add(location=loc);cam=bpy.context.object;cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=a.ortho_scale*(1+max(0.,a.explode));cam.data.clip_start=.01;cam.data.clip_end=20
scene=bpy.context.scene;scene.camera=cam;scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.3
for loc,energy,size in [((2,-3,4),40,3),((-3,-1,2),25,2),((1,3,3),30,2)]:
 bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.rotation_euler=(-o.location).to_track_quat('-Z','Y').to_euler();o.data.energy=energy;o.data.size=size
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.render.resolution_x=768;scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.view_settings.view_transform='Standard';scene.render.film_transparent=False
out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(out)
bpy.context.view_layer.update()
(out.parent/(out.stem+'_camera.json')).write_text(json.dumps({'view':a.view,'matrix_world':[list(row) for row in cam.matrix_world],'matrix_world_inverse':[list(row) for row in cam.matrix_world.inverted()],'type':'ORTHO','ortho_scale':cam.data.ortho_scale,'resolution':[768,768],'target':[0,0,0],'normalization':'none (pre-normalized geometry)','obj_axis_import':{'forward':'Y','up':'Z'}},indent=2))
bpy.ops.render.render(write_still=True)
