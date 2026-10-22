"""Protect the existing visual evidence and exports while removing repeated work."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image, ImageDraw

from agent_runner import publish_round, save_round
from api_client import Client
from render_preview import _panel, _projection, render_views
from voxel_engine import VoxelCanvas


def original_panel(coords, colors, view, canvas_size=336):
    """Original 4d7ca8f painter, including overlapping rectangles and depth ties."""
    im=Image.new('RGB',(canvas_size,canvas_size),'#f8fafc');draw=ImageDraw.Draw(im)
    corners=np.array([[a,b,c] for a in (0,63) for b in (0,63) for c in (0,63)],float)
    uc,vc,_=_projection(corners,view)
    lo_u,hi_u,lo_v,hi_v=np.min(uc),np.max(uc),np.min(vc),np.max(vc)
    scale=min((canvas_size-38)/(hi_u-lo_u+4),(canvas_size-38)/(hi_v-lo_v+4))
    off_u=(canvas_size-scale*(hi_u-lo_u))/2;off_v=(canvas_size-scale*(hi_v-lo_v))/2
    if len(coords):
        u,v,depth=_projection(coords,view);rad=max(1,int(scale*.72))
        for i in np.argsort(depth):
            sx=int((u[i]-lo_u)*scale+off_u);sy=int((v[i]-lo_v)*scale+off_v)
            draw.rectangle((sx-rad,sy-rad,sx+rad,sy+rad),fill=tuple(int(c) for c in colors[i]))
    draw.text((10,9),view,fill='#263446')
    return im


@pytest.mark.parametrize('view',['Front (-Y)','Side (+X)','Top (+Z)','Isometric'])
def test_render_matches_old_painter_with_occlusion_depth_ties_and_empty_scene(view):
    rng=np.random.default_rng(9823)
    xyz=rng.integers(0,64,size=(12000,3)).astype(float)
    # Exact coordinate duplicates with different colors must retain the original tie order.
    xyz[6000:]=xyz[:6000]
    colors=rng.integers(0,256,size=(12000,3),dtype=np.uint8)
    for points,paint in [(xyz,colors),(xyz[:0],colors[:0])]:
        np.testing.assert_array_equal(np.asarray(_panel(points,paint,view)),
                                      np.asarray(original_panel(points,paint,view)))


def test_published_round_matches_independent_export_and_render(tmp_path):
    canvas=VoxelCanvas();canvas.apply({'op':'brush','part':'body','points':[[10,10,10],[30,30,30]],'radius':2})
    graph={'parts':[{'id':'body','parent':None}]}
    reference=tmp_path/'ref.png';Image.new('RGB',(8,8),'white').save(reference)
    rd=tmp_path/'round';old=tmp_path/'old';out=tmp_path/'published';out.mkdir()
    save_round(canvas,graph,rd,reference);render_views(canvas,rd/'views')
    save_round(canvas,graph,old,reference);render_views(canvas,old/'views_final')
    (out/'ind_99.npy').write_bytes(b'stale');(out/'agent_status.json').write_text('keep')
    publish_round(rd,out)
    assert not (out/'ind_99.npy').exists()
    assert (out/'agent_status.json').read_text()=='keep'
    for p in old.rglob('*'):
        if p.is_file():assert p.read_bytes()==(out/p.relative_to(old)).read_bytes()
    # Publishing must copy, not hard-link artifacts that might later be overwritten.
    (out/'summary.json').write_text('changed')
    assert (rd/'summary.json').read_text()!='changed'


def test_client_reuses_real_http_connection_and_closes_session(tmp_path,monkeypatch):
    ports=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            ports.append(self.client_address[1])
            payload=json.dumps({'type':'response.completed','response':{'model':'test-model','output':[]}})
            body=('data: '+payload+'\n\n').encode()
            self.send_response(200);self.send_header('Content-Type','text/event-stream')
            self.send_header('Content-Length',str(len(body)));self.end_headers()
            self.wfile.write(body);self.wfile.flush()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    config=tmp_path/'api.json'
    config.write_text(json.dumps({'protocol':'responses','base_url':f'http://127.0.0.1:{server.server_port}',
                                  'api_key':'test-key'}))
    client=Client(config,'test-model',tmp_path/'logs');client.session.trust_env=False
    try:
        client.call([],'first');client.call([],'second')
        assert len(ports)==2 and ports[0]==ports[1]
        closed=[];close=client.session.close
        def record_close():closed.append(True);close()
        monkeypatch.setattr(client.session,'close',record_close)
        client.close();assert closed==[True]
    finally:
        client.close();server.shutdown();server.server_close();worker.join(timeout=2)
