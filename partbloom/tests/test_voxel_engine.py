import numpy as np
from voxel_engine import VoxelCanvas


def test_polygon_and_erase():
    v=VoxelCanvas()
    v.apply({'op':'polygon','part':'seat','plane':'xy','start':20,'end':21,'vertices':[[10,10],[20,10],[20,20],[10,20]]})
    assert v.parts['seat'][15,15,20]
    assert v.parts['seat'][15,15,21]
    assert not v.parts['seat'][15,15,22]
    v.apply({'op':'polygon','part':'seat','plane':'xy','start':20,'end':20,'vertices':[[10,10],[20,10],[20,20],[10,20]],'mode':'erase'})
    assert not v.parts['seat'][15,15,20]
    assert v.parts['seat'][15,15,21]


def test_brush_translate():
    v=VoxelCanvas()
    v.apply({'op':'brush','part':'wheel','points':[[10,10,10],[10,10,16]],'radius':2,'mode':'add'})
    assert v.parts['wheel'][10,10,13]
    v.apply({'op':'move','part':'wheel','delta':[4,0,0]})
    assert v.parts['wheel'][14,10,13]
    assert not v.parts['wheel'][10,10,13]


def test_export_official_compatible_shape_and_order(tmp_path):
    v=VoxelCanvas()
    for k in (0,1):
        v.apply({'op':'polygon','part':f'part{k}','plane':'xy','start':5+k*12,'end':7+k*12,
                 'vertices':[[7,7],[17,7],[17,17],[7,17]]})
    result=v.export(tmp_path)
    assert result['parts']==2
    a=np.load(tmp_path/'allind.npy')
    b=np.concatenate([np.load(tmp_path/'ind_0.npy'),np.load(tmp_path/'ind_1.npy')])
    assert np.array_equal(a,b)
    assert a.ndim==2 and a.shape[1]==3
    assert a.dtype==np.int64
    assert a.min()>=0 and a.max()<64
    dense = np.load(tmp_path/'dense_parts.npz')['volumes']
    assert dense.shape==(2,64,64,64)
    assert len(a)<dense.sum()
