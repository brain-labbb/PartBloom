"""Pure-Pillow multi-view voxel renderer, including a simple isometric view.

This renderer deliberately requires no GPU, CUDA, display, Blender or OpenGL.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import binary_erosion, generate_binary_structure

COLORS = [
    (68, 123, 204), (230, 142, 77), (71, 172, 130),
    (176, 119, 205), (225, 193, 88), (107, 173, 191),
    (213, 111, 125), (133, 147, 171)
]


def _scene_points(canvas):
    xyz, col = [], []
    for i, (_, volume) in enumerate(canvas.parts.items()):
        surface = volume & ~binary_erosion(volume, generate_binary_structure(3, 1), border_value=0)
        coords = np.argwhere(surface)
        xyz.append(coords)
        col.extend([COLORS[i % len(COLORS)]] * len(coords))
    if not xyz or sum(len(c) for c in xyz) == 0:
        return np.zeros((0,3)), np.zeros((0,3), dtype=np.uint8)
    return np.concatenate(xyz, 0).astype(float), np.array(col, dtype=np.uint8)


def _projection(p, view):
    x,y,z = p.T
    if view == 'Front (-Y)':
        return x, -z, -y
    if view == 'Side (+X)':
        return y, -z, x
    if view == 'Top (+Z)':
        return x, y, z
    if view == 'Isometric':
        return 0.76*(x+y), 0.32*(x-y)-0.97*z, 0.85*x-0.85*y+0.5*z
    raise ValueError(view)


def _panel(coords, colors, view, canvas_size=336):
    im = Image.new('RGB', (canvas_size, canvas_size), '#f8fafc')
    d = ImageDraw.Draw(im)
    corners = np.array([[a,b,c] for a in (0,63) for b in (0,63) for c in (0,63)],float)
    uc,vc,_ = _projection(corners, view)
    lo_u,hi_u,lo_v,hi_v = np.min(uc),np.max(uc),np.min(vc),np.max(vc)
    scale = min((canvas_size-38)/(hi_u-lo_u+4), (canvas_size-38)/(hi_v-lo_v+4))
    off_u = (canvas_size-scale*(hi_u-lo_u))/2
    off_v = (canvas_size-scale*(hi_v-lo_v))/2
    if len(coords):
        u,v,depth = _projection(coords,view)
        order=np.argsort(depth)
        rad=max(1,int(scale*0.72))
        screen=np.column_stack(((u[order]-lo_u)*scale+off_u,
                                (v[order]-lo_v)*scale+off_v)).astype(int)
        # Equal centers draw identical rectangles: only the LAST survives.
        # Preserve the original order of surviving rectangles, including depth ties.
        _, reverse_indices=np.unique(screen[::-1],axis=0,return_index=True)
        keep=np.sort(len(order)-1-reverse_indices)
        for pos in keep:
            sx,sy=map(int,screen[pos])
            color=tuple(int(c) for c in colors[order[pos]])
            d.rectangle((sx-rad,sy-rad,sx+rad,sy+rad), fill=color)
    d.text((10,9),view,fill='#263446')
    return im


def render_views(canvas, output_dir):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    xyz,col=_scene_points(canvas)
    view_names=['Front (-Y)','Side (+X)','Top (+Z)','Isometric']
    ims=[]
    for name in view_names:
        img=_panel(xyz,col,name)
        img.save(out/(name.split(' ')[0].lower()+'.png'))
        ims.append(img)
    collage=Image.new('RGB',(672,672),'white')
    for i,img in enumerate(ims):
        collage.paste(img,((i%2)*336,(i//2)*336))
    collage.save(out/'preview.png')
    return str(out/'preview.png')
