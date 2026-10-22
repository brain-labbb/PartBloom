"""Prepare a new decode experiment from any existing shared SLat export."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['joint','parts'],default='parts')
    a=p.parse_args();source=a.source.resolve();out=a.out.resolve()
    latent=source/'shared_slat.pt'
    digest=hashlib.sha256(latent.read_bytes()).hexdigest()
    out.mkdir(parents=True,exist_ok=False)
    for name in ['allind.npy','part_labels.npy','part_index.json','cond_img.png','basic_info.txt','part_graph.json']:
        if (source/name).exists():shutil.copy2(source/name,out/name)
    for file in source.glob('ind_*.npy'):shutil.copy2(file,out/file.name)
    (out/'experiment.json').write_text(json.dumps({'mode':a.mode,'latent_source':str(latent),
        'latent_sha256':digest,'support_sha256':hashlib.sha256((out/'allind.npy').read_bytes()).hexdigest(),
        'baseline':str(source),'changed_factor':'mesh decoder context only',
        'labels':'identical source labels','mesh_inner_visibility':.01},indent=2))
    print(out)


if __name__=='__main__':main()
