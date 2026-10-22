"""Preview restored TRELLIS textures from standard Y-up GLB; display-only normalization."""
import argparse,sys,json
from pathlib import Path
import bpy
from mathutils import Vector
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--view',default='oblique',choices=['oblique','front','side','top']);p.add_argument('--ortho-scale',type=float,default=1.35);a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(Path(a.input).resolve()))
from mathutils import Matrix
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
if not objects:raise RuntimeError('No textured meshes to render')
points=[o.matrix_world@Vector(c) for o in objects for c in o.bound_box]
lo=Vector([min(p[i] for p in points) for i in range(3)])
hi=Vector([max(p[i] for p in points) for i in range(3)])
center=(lo+hi)/2;scale=1/max(hi-lo)
# Standard Y-up GLB imports into Blender's Z-up automatically.
transform=Matrix.Scale(scale,4)@Matrix.Translation(-center)
matrices={o:transform@o.matrix_world for o in objects}
for o,m in matrices.items():o.parent=None;o.matrix_world=m
bpy.context.view_layer.update()
objects=[o for o in bpy.context.scene.objects if o.type=='MESH']
if not objects:raise RuntimeError('No meshes to render')
# Preserve embedded native TRELLIS base-color textures.
loc={'oblique':(1.5,-2.5,1.5),'front':(0,-3,0),'side':(3,0,0),'top':(0,0,3)}[a.view]
bpy.ops.object.camera_add(location=loc);cam=bpy.context.object;cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=a.ortho_scale;cam.data.clip_start=.01;cam.data.clip_end=20
scene=bpy.context.scene;scene.camera=cam;scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(1,1,1,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.8
for loc,energy,size in [((2,-3,4),400,3),((-3,-1,2),250,2),((1,3,3),300,2)]:
 bpy.ops.object.light_add(type='AREA',location=loc);o=bpy.context.object;o.rotation_euler=(-o.location).to_track_quat('-Z','Y').to_euler();o.data.energy=energy;o.data.size=size
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24;scene.cycles.use_denoising=True;scene.render.resolution_x=768;scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB';scene.view_settings.view_transform='Standard';scene.render.film_transparent=False
out=Path(a.output).resolve();out.parent.mkdir(parents=True,exist_ok=True);scene.render.filepath=str(out)
bpy.context.view_layer.update()
(out.parent/(out.stem+'_camera.json')).write_text(json.dumps({'view':a.view,'matrix_world':[list(row) for row in cam.matrix_world],'matrix_world_inverse':[list(row) for row in cam.matrix_world.inverted()],'type':'ORTHO','ortho_scale':cam.data.ortho_scale,'resolution':[768,768],'target':[0,0,0],'normalization':{'center':list(center),'scale':scale,'display_only':True},'input_axes':'standard glTF Y-up imported as Blender Z-up'},indent=2))
bpy.ops.render.render(write_still=True)
