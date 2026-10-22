"""Real vision + function-calling agent for direct 64^3 voxel editing.

Usage:
    OPENAI_API_KEY=... python agent_runner.py --image chair.png --model gpt-6.1-sol

Requires an API endpoint supporting the OpenAI Responses API,
vision inputs, and function calling. This script does NOT run TRELLIS;
it exports the exact file naming convention expected by PhysX-Omni.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import shutil
from pathlib import Path

import time
import traceback
import numpy as np
import re
from api_client import Client, response_text, parse_json
from voxel_engine import VoxelCanvas
from render_preview import render_views

from prompts.definitions import (TOOLS, COORDINATE_INSTRUCTIONS, PLANNER_INSTRUCTIONS,
                                 BUILDER_INSTRUCTIONS, REVIEWER_INSTRUCTIONS)


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError(f"Supported reference image types: PNG, JPEG, WebP; got {mime}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def img_content(path: Path, detail: str = "auto") -> dict:
    return {"type": "input_image", "image_url": image_data_url(path), "detail": detail}


def describe_state(canvas: VoxelCanvas) -> str:
    return json.dumps(canvas.summary(), ensure_ascii=False)


def dispatch_tool(canvas: VoxelCanvas, name: str, args: dict, allowed_parts: set | None = None) -> dict:
    if name == "inspect_voxel_state":
        return canvas.summary()
    if name != "apply_voxel_edits":
        raise ValueError("Unknown function: " + name)
    edits = args.get('edits', [])
    if not 1 <= len(edits) <= 15:
        raise ValueError("Pass 1..15 edits per call")
    results = []
    for edit in edits:
        try:
            if allowed_parts is not None and edit.get("part") not in allowed_parts:
                raise ValueError("Only edit the assigned planned part(s)")
            results.append({"ok": True, **canvas.apply(edit)})
        except (ValueError, TypeError, IndexError, KeyError) as exc:
            results.append({"ok": False, "error": str(exc), "edit": edit})
    return {"results": results, "state": canvas.summary()}


def run_builder(client: Client, model: str, canvas: VoxelCanvas, image_path: Path,
                out: Path, round_no: int, feedback: dict | None = None,
                previous_preview: Path | None = None, max_api_steps: int = 10,
                graph: dict | None = None, part_id: str | None = None) -> None:
    text = (
        f"Builder round={round_no}. "
        "Create a recognizable 3D structural voxel proxy of the image. "
        "Actually CALL apply_voxel_edits; you are editing a persistent voxel canvas. "
        "Aim for the major semantic parts and approximate proportion. "
        f"Current state={describe_state(canvas)}"
    )
    if graph is not None:
        text += "\nImage-derived part plan=" + json.dumps(graph)
    if part_id is not None:
        text += "\nBuild ONLY part " + part_id + ". Other parts are handled separately."
    if feedback is not None:
        text += "\nReviewer identified these issues. Fix them LOCALLY, not by rebuilding everything: " + json.dumps(feedback, ensure_ascii=False)
    text += (f"\nYou have at most {max_api_steps} model requests for this stage, including your final response. "
             "This is a ceiling, not a target. Finish early with a text response without tool calls when done.")
    content = [{"type":"input_text", "text":text}, img_content(image_path)]
    if previous_preview is not None:
        content.extend([{"type":"input_text", "text":"Here is the current four-view voxel render (colored by semantic part)."},
                        img_content(previous_preview)])
    inputs = [{"role":"user", "content":content}]
    trace=[]
    for step in range(max_api_steps):
        label=f"builder_round{round_no}_{part_id or 'repair'}_step{step}"
        response = client.call([{"role":"developer","content":BUILDER_INSTRUCTIONS}]+inputs, label, TOOLS)
        inputs.extend(response['output'])
        calls=[item for item in response['output'] if item['type'] == 'function_call']
        if not calls:
            trace.append({"assistant_final": response_text(response)})
            required = {part_id} if part_id else ({p['id'] for p in graph['parts']} if graph else set())
            empty = sorted(pid for pid in required if not canvas.parts.get(pid, np.zeros(0)).any())
            unexpected = sorted(set(canvas.parts) - required) if graph and not part_id else []
            if empty or unexpected:
                feedback_text = (
                    'The stage is NOT complete. Canvas validation failed: '
                    + json.dumps({'empty_or_missing_parts': empty, 'unexpected_parts': unexpected})
                    + '. The apply_voxel_edits tool IS available in this request. '
                    'Your prose or JSON text does not execute edits. Actually call the supplied tool '
                    'to create the missing planned geometry. Use only the supported operations and '
                    'check tool results. Do not claim completion until the assigned parts are nonempty.'
                )
                trace.append({'validation_feedback': feedback_text})
                inputs.append({'role': 'user', 'content': [{'type': 'input_text', 'text': feedback_text}]})
                continue
            break
        for call in calls:
            try:
                args=json.loads(call["arguments"])
                allowed={part_id} if part_id else ({p["id"] for p in graph["parts"]} if graph else None)
                result=dispatch_tool(canvas, call["name"], args, allowed_parts=allowed)
            except Exception as exc:
                result={"ok":False,"error":str(exc)}
            trace.append({"tool": call["name"], "arguments": args if 'args' in locals() else {}, "output": result})
            inputs.append({"type":"function_call_output", "call_id":call["call_id"],
                           "output":json.dumps(result, ensure_ascii=False)})
    else:
        (out/f"builder_round{round_no}_{part_id or 'repair'}.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf8')
        raise RuntimeError("Builder exceeded maximum API steps")
    path = out/f"builder_round{round_no}_{part_id or 'repair'}.json"
    path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf8")


def run_reviewer(client: Client, model: str, reference: Path, preview: Path, canvas: VoxelCanvas,
                 out: Path, round_no: int) -> dict:
    inputs=[{"role":"user", "content":[
        {"type":"input_text", "text":"Image 1 is the TARGET object."}, img_content(reference),
        {"type":"input_text", "text":"Image 2 is the current Voxel model, four projected views. Review its shape and parts only."},
        img_content(preview),
        {"type":"input_text", "text":"Voxel state="+describe_state(canvas)}
    ]}]
    response=client.call([{"role":"developer","content":REVIEWER_INSTRUCTIONS}]+inputs, f"review_round{round_no}")
    parsed=parse_json(response_text(response))
    if not isinstance(parsed.get('issues'), list) or len(parsed['issues'])>3:
        raise ValueError('Reviewer must return at most 3 issues')
    (out/f"review_round{round_no}.json").write_text(json.dumps(parsed,ensure_ascii=False,indent=2),encoding='utf8')
    return parsed


def validate_graph(graph):
    parts=graph['parts'];ids=[p['id'] for p in parts]
    if not parts or len(ids)!=len(set(ids)):
        raise ValueError('Need nonempty unique parts')
    for part in parts:
        if not re.fullmatch(r"[A-Za-z][\w-]{0,39}",part['id']):raise ValueError('Invalid part ID')
        if 'joint_type' in part and part['joint_type'] not in ['fixed','revolute','prismatic','unknown']:raise ValueError('Invalid joint type')
        seen={part['id']};parent=part['parent']
        while parent is not None:
            if parent not in ids or parent in seen:raise ValueError('Invalid or cyclic parent')
            seen.add(parent);parent=parts[ids.index(parent)]['parent']
    return graph


def plan_parts(client,reference,out):
    response=client.call([{'role':'user','content':[{'type':'input_text','text':PLANNER_INSTRUCTIONS},img_content(reference)]}],'part_planning')
    graph=validate_graph(parse_json(response_text(response)))
    (out/'part_graph.json').write_text(json.dumps(graph,indent=2))
    return graph


def save_round(canvas,graph,out,reference):
    ids=[p['id'] for p in graph['parts']]
    if set(canvas.parts)!=set(ids):raise ValueError('Canvas parts differ from plan')
    if any(not canvas.parts[pid].any() for pid in ids):raise ValueError('Planned part is empty')
    canvas.parts={pid:canvas.parts[pid] for pid in ids}
    result=canvas.export(out,reference)
    index=json.loads((out/'part_index.json').read_text())
    if [x['part'] for x in index]!=ids:raise ValueError('Surface export lost a planned part')
    (out/'part_graph.json').write_text(json.dumps(graph,indent=2))
    return result


def publish_round(round_dir, out):
    """Publish the already exported round without repeating export or rendering."""
    for stale in out.glob('ind_*.npy'):
        stale.unlink()
    for artifact in round_dir.iterdir():
        if artifact.is_file():
            shutil.copy2(artifact, out / artifact.name)
    shutil.copytree(round_dir / 'views', out / 'views_final', dirs_exist_ok=True)


def main():
    p=argparse.ArgumentParser(description="Image-only part planning, Builder tools and Reviewer, then official voxel export")
    p.add_argument('--image',required=True);p.add_argument('--out',default='outputs/B1_sample')
    p.add_argument('--model',default='gpt-6.1-sol');p.add_argument('--repairs',type=int,default=2,choices=[0,1,2])
    p.add_argument('--reasoning-effort',choices=['low','medium','high'],default='high')
    p.add_argument('--max-api-steps',type=int,default=10)
    p.add_argument('--repair-max-api-steps',type=int,default=20)
    p.add_argument('--api-config',type=Path,required=True)
    p.add_argument('--protocol', choices=['responses', 'generate_content', 'anthropic', 'chat_completions'])
    args=p.parse_args();reference=Path(args.image).resolve();out=Path(args.out).resolve()
    if args.max_api_steps < 1 or args.repair_max_api_steps < 1:
        p.error('Build and repair request limits must be positive')
    if not reference.is_file():p.error('Missing RGB input')
    if (out/'part_graph.json').exists():p.error('Output already contains a run; use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'prompt_protocol.json').write_text(json.dumps({
        'version':'functional_motion_voxel_tools_v3','representation':'direct voxel editing; no Agent RLE',
        'planner':PLANNER_INSTRUCTIONS,'builder':BUILDER_INSTRUCTIONS,
        'reviewer':REVIEWER_INSTRUCTIONS,'tools':TOOLS,
        'budgets':{'build_max_api_steps':args.max_api_steps,'repair_max_api_steps':args.repair_max_api_steps,
                   'max_edits_per_call':15}},indent=2))
    client=Client(args.api_config,args.model,out/'api_logs',reasoning_effort=args.reasoning_effort,protocol=args.protocol);canvas=VoxelCanvas();start=time.time()
    status={'model':args.model,'reasoning_effort':args.reasoning_effort,'repairs_limit':args.repairs,'input':str(reference),'cost_currency':None,'cost_note':'No verified provider unit prices; raw token usage saved','status':'running'}
    (out/'agent_status.json').write_text(json.dumps(status,indent=2))
    try:
        graph=plan_parts(client,reference,out)
        for part in graph['parts']:
            canvas._part(part['id'])
        for part in graph['parts']:
            run_builder(client,args.model,canvas,reference,out,0,max_api_steps=args.max_api_steps,graph=graph,part_id=part['id'])
            if not canvas.parts.get(part['id'],np.zeros(0)).any():raise ValueError('Builder left empty part')
        for r in range(args.repairs+1):
            rd=out/'rounds'/f'round{r}';rd.mkdir(parents=True,exist_ok=True)
            save_round(canvas,graph,rd,reference);preview=Path(render_views(canvas,rd/'views'))
            publish_round(rd,out)
            review=run_reviewer(client,args.model,reference,preview,canvas,out,r)
            if not review['issues'] or r==args.repairs:
                status['repair_rounds']=r;status['final_reviewer']=review;break
            run_builder(client,args.model,canvas,reference,out,r+1,feedback=review,previous_preview=preview,max_api_steps=args.repair_max_api_steps,graph=graph)
        status['status']='complete';status['part_count']=len(graph['parts'])
    except Exception:
        status['status']='failed';status['traceback']=traceback.format_exc();raise
    finally:
        status['seconds']=time.time()-start;status['api_calls']=client.calls
        status['api_seconds']=sum(call['seconds'] for call in client.calls)
        status['non_api_seconds']=max(0,status['seconds']-status['api_seconds'])
        status['harness_version']='v3_io_optimized_baseline_restored'
        (out/'agent_status.json').write_text(json.dumps(status,indent=2))
        client.close()

if __name__=='__main__':
    main()
