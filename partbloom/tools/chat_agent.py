"""Chat-agent bridge for the PartBloom coarse stage.

Replaces the API Client of pipeline/agent_runner.py with a command-line tool
that an interactive agent (e.g. a Claude Code subagent) calls directly. The
canvas, tool executor, prompts, budgets, export and preview are the mainline
ones; only the transport differs. Each `edit`, `inspect` or `done` invocation
counts as one model request toward the same per-stage limits.

Typical sequence for one image (RUN is the per-image output folder):
    init --image IMG --out RUN
    prompt --stage planner            -> write plan JSON -> plan --run RUN --graph plan.json
    for each part in plan order:
        prompt --stage builder --run RUN --part P
        edit --run RUN --part P --edits edits.json   (repeat; 1..15 edits each)
        done --run RUN --part P --text "final remarks"
    snapshot --run RUN                -> view preview.png
    prompt --stage reviewer           -> review --run RUN --json review.json
    if issues and repairs remain:
        prompt --stage builder --run RUN --repair
        edit --run RUN --repair --edits edits.json ... ; done --run RUN --repair --text ...
        snapshot / review again
    (review finalises the coarse stage automatically when it ends the loop)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE), str(PACKAGE / 'pipeline')]

import numpy as np  # noqa: E402
from agent_runner import dispatch_tool, validate_graph, save_round, publish_round  # noqa: E402
from api_client import parse_json  # noqa: E402
from render_preview import render_views  # noqa: E402
from voxel_engine import VoxelCanvas  # noqa: E402
from prompts.definitions import (TOOLS, PLANNER_INSTRUCTIONS, BUILDER_INSTRUCTIONS,  # noqa: E402
                                 REVIEWER_INSTRUCTIONS)

REPAIRS = 2
BUILD_STEPS = 10
REPAIR_STEPS = 20


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


class Run:
    def __init__(self, run_dir):
        self.root = Path(run_dir).resolve()
        self.coarse = self.root / 'coarse'
        self.chat = self.coarse / '_chat'
        if not (self.chat / 'state.json').is_file():
            raise SystemExit(f'Not an initialised chat run: {self.root}')
        self.state = load(self.chat / 'state.json')

    def save_state(self):
        dump(self.chat / 'state.json', self.state)

    def canvas(self):
        canvas = VoxelCanvas.from_npz(self.chat / 'canvas.npz')
        canvas.action_log = load(self.chat / 'action_log.json')
        order = load(self.chat / 'canvas_order.json')
        canvas.parts = {name: canvas.parts[name] for name in order}
        return canvas

    def save_canvas(self, canvas):
        names = list(canvas.parts)
        volumes = np.stack([canvas.parts[n] for n in names]) if names else np.zeros((0, 64, 64, 64), bool)
        np.savez_compressed(self.chat / 'canvas.npz', names=np.array(names), volumes=volumes)
        dump(self.chat / 'canvas_order.json', names)
        dump(self.chat / 'action_log.json', canvas.action_log)

    def status(self, **updates):
        path = self.coarse / 'agent_status.json'
        status = load(path)
        status.update(updates)
        status['seconds'] = time.time() - self.state['started']
        dump(path, status)
        return status

    def stage(self, part, repair):
        """Return (stage key, request limit, allowed parts, trace path) for the active builder stage."""
        graph = self.state.get('graph')
        if graph is None:
            raise SystemExit('Plan the parts first')
        ids = [p['id'] for p in graph['parts']]
        if repair:
            if self.state['phase'] != 'repair':
                raise SystemExit(f"No repair stage is open (phase={self.state['phase']})")
            r = self.state['round']
            return f'round{r}_repair', REPAIR_STEPS, set(ids), self.coarse / f'builder_round{r}_repair.json'
        if self.state['phase'] != 'build':
            raise SystemExit(f"Initial part building is closed (phase={self.state['phase']})")
        expected = ids[len(self.state['built'])]
        if part != expected:
            raise SystemExit(f'Build parts in plan order; current part is {expected}')
        return f'round0_{part}', BUILD_STEPS, {part}, self.coarse / f'builder_round0_{part}.json'

    def request(self, key, limit, trace_path):
        used = self.state['requests'].get(key, 0)
        if used >= limit:
            raise SystemExit(f'Stage {key} has used all {limit} requests; finish with `done`')
        self.state['requests'][key] = used + 1
        self.state['request_total'] = self.state.get('request_total', 0) + 1
        trace = load(trace_path) if trace_path.exists() else []
        return used + 1, trace


def cmd_init(a):
    image = Path(a.image).resolve()
    root = Path(a.out).resolve()
    if root.exists():
        raise SystemExit(f'Output already exists: {root}. Choose a new folder.')
    from PIL import Image
    root.mkdir(parents=True)
    rgba = Image.open(image).convert('RGBA')
    # Composite RGBA on white, matching run.py's full-run preprocessing.
    Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB').save(root / 'input.png')
    config = json.loads((PACKAGE / 'configs/default.json').read_text())
    config.update(model=a.model, reasoning_effort=a.reasoning_effort, protocol=None,
                  restore_missing_parts=True)
    dump(root / 'run.json', {
        'image': str(image), 'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
        'config': config, 'prompt_version': 'functional_motion_voxel_tools_v3',
        'agent_transport': 'chat_agent_bridge', 'agent_identity': a.agent_identity})
    coarse = root / 'coarse'
    chat = coarse / '_chat'
    chat.mkdir(parents=True)
    dump(coarse / 'prompt_protocol.json', {
        'version': 'functional_motion_voxel_tools_v3', 'representation': 'direct voxel editing; no Agent RLE',
        'planner': PLANNER_INSTRUCTIONS, 'builder': BUILDER_INSTRUCTIONS,
        'reviewer': REVIEWER_INSTRUCTIONS, 'tools': TOOLS,
        'budgets': {'build_max_api_steps': BUILD_STEPS, 'repair_max_api_steps': REPAIR_STEPS,
                    'max_edits_per_call': 15},
        'transport': 'chat_agent_bridge: tool calls issued via tools/chat_agent.py, no API client'})
    started = time.time()
    dump(coarse / 'agent_status.json', {
        'model': a.model, 'reasoning_effort': a.reasoning_effort, 'repairs_limit': REPAIRS,
        'input': str(root / 'input.png'), 'agent_identity': a.agent_identity,
        'transport': 'chat_agent_bridge', 'cost_currency': None,
        'cost_note': 'Interactive chat agent; no API token usage recorded by the harness',
        'status': 'running', 'api_calls': [], 'harness_version': 'v3_chat_agent_bridge'})
    state = {'phase': 'plan', 'round': 0, 'built': [], 'requests': {}, 'started': started,
             'input': str(root / 'input.png'), 'graph': None, 'snapshots': [], 'reviews': []}
    dump(chat / 'state.json', state)
    run = Run(root)
    run.save_canvas(VoxelCanvas())
    print(json.dumps({'run': str(root), 'reference_image': str(root / 'input.png'),
                      'next': 'prompt --stage planner'}, indent=2))


def cmd_prompt(a):
    if a.stage == 'planner':
        print(PLANNER_INSTRUCTIONS)
        print('\nView the reference image, then write the JSON plan to a file and call `plan`.')
        return
    if a.stage == 'reviewer':
        print(REVIEWER_INSTRUCTIONS)
        print('\nImage 1 is the TARGET object (reference). Image 2 is the current voxel model preview '
              'printed by `snapshot`. Review its shape and parts only.')
        return
    run = Run(a.run)
    canvas = run.canvas()
    graph = run.state['graph']
    if graph is None:
        raise SystemExit('Plan the parts first')
    key, limit, _, _ = run.stage(a.part, a.repair)
    r = run.state['round']
    text = (f'Builder round={0 if not a.repair else r}. '
            'Create a recognizable 3D structural voxel proxy of the image. '
            'Actually CALL apply_voxel_edits; you are editing a persistent voxel canvas. '
            'Aim for the major semantic parts and approximate proportion. '
            f'Current state={json.dumps(canvas.summary(), ensure_ascii=False)}'
            '\nImage-derived part plan=' + json.dumps(graph))
    if not a.repair:
        text += '\nBuild ONLY part ' + a.part + '. Other parts are handled separately.'
    else:
        review = load(run.coarse / f'review_round{r - 1}.json')
        text += ('\nReviewer identified these issues. Fix them LOCALLY, not by rebuilding everything: '
                 + json.dumps(review, ensure_ascii=False))
        text += f"\nThe current four-view voxel render (colored by semantic part): {run.state['snapshots'][-1]}"
    used = run.state['requests'].get(key, 0)
    text += (f'\nYou have at most {limit} model requests for this stage, including your final response '
             f'(already used {used}). This is a ceiling, not a target. Finish early with `done` when done.')
    print(BUILDER_INSTRUCTIONS)
    print('\n--- stage input ---\n' + text)
    print(f"\nReference image: {run.state['input']}")
    print('Tool apply_voxel_edits => `edit --edits FILE` where FILE holds {"edits":[...]} or a list of 1..15 edits.')


def cmd_plan(a):
    run = Run(a.run)
    if run.state['phase'] != 'plan':
        raise SystemExit('Plan already recorded')
    graph = validate_graph(parse_json(Path(a.graph).read_text(encoding='utf8')))
    canvas = run.canvas()
    for part in graph['parts']:
        canvas._part(part['id'])
    dump(run.coarse / 'part_graph.json', graph)
    run.state.update(graph=graph, phase='build')
    run.state['request_total'] = run.state.get('request_total', 0) + 1
    run.save_canvas(canvas)
    run.save_state()
    run.status(planned_parts=[p['id'] for p in graph['parts']])
    print(json.dumps({'parts': [p['id'] for p in graph['parts']],
                      'next': f"build part {graph['parts'][0]['id']}"}, indent=2))


def cmd_edit(a, tool):
    run = Run(a.run)
    key, limit, allowed, trace_path = run.stage(a.part, a.repair)
    step, trace = run.request(key, limit, trace_path)
    canvas = run.canvas()
    args = {}
    if tool == 'apply_voxel_edits':
        payload = json.loads(Path(a.edits).read_text(encoding='utf8'))
        args = payload if isinstance(payload, dict) else {'edits': payload}
    try:
        result = dispatch_tool(canvas, tool, args, allowed_parts=allowed)
    except Exception as exc:
        result = {'ok': False, 'error': str(exc)}
    trace.append({'tool': tool, 'arguments': args, 'output': result})
    dump(trace_path, trace)
    run.save_canvas(canvas)
    run.save_state()
    print(json.dumps({'request': f'{step}/{limit}', **result}, ensure_ascii=False))


def cmd_done(a):
    run = Run(a.run)
    key, limit, _, trace_path = run.stage(a.part, a.repair)
    step, trace = run.request(key, limit, trace_path)
    if not a.repair:
        canvas = run.canvas()
        if not canvas.parts.get(a.part, np.zeros(0)).any():
            run.save_state()
            raise SystemExit('Builder left empty part; add voxels before finishing')
    trace.append({'assistant_final': a.text})
    dump(trace_path, trace)
    if a.repair:
        run.state['phase'] = 'snapshot'
    else:
        run.state['built'].append(a.part)
        if len(run.state['built']) == len(run.state['graph']['parts']):
            run.state['phase'] = 'snapshot'
    run.save_state()
    nxt = 'snapshot' if run.state['phase'] == 'snapshot' else \
        f"build part {run.state['graph']['parts'][len(run.state['built'])]['id']}"
    print(json.dumps({'request': f'{step}/{limit}', 'stage_finished': key, 'next': nxt}, indent=2))


def cmd_inspect(a):
    cmd_edit(a, 'inspect_voxel_state')


def cmd_snapshot(a):
    run = Run(a.run)
    if run.state['phase'] != 'snapshot':
        raise SystemExit(f"Snapshot not expected now (phase={run.state['phase']})")
    canvas = run.canvas()
    r = run.state['round']
    rd = run.coarse / 'rounds' / f'round{r}'
    rd.mkdir(parents=True, exist_ok=True)
    save_round(canvas, run.state['graph'], rd, Path(run.state['input']))
    preview = Path(render_views(canvas, rd / 'views'))
    publish_round(rd, run.coarse)
    run.save_canvas(canvas)
    run.state['snapshots'].append(str(preview))
    run.state['phase'] = 'review'
    run.save_state()
    print(json.dumps({'round': r, 'preview': str(preview), 'reference': run.state['input'],
                      'state': canvas.summary(), 'next': 'prompt --stage reviewer; then review'},
                     ensure_ascii=False, indent=2))


def cmd_review(a):
    run = Run(a.run)
    if run.state['phase'] != 'review':
        raise SystemExit(f"Review not expected now (phase={run.state['phase']})")
    parsed = parse_json(Path(a.json).read_text(encoding='utf8'))
    if not isinstance(parsed.get('issues'), list) or len(parsed['issues']) > 3:
        raise SystemExit('Reviewer must return at most 3 issues')
    r = run.state['round']
    dump(run.coarse / f'review_round{r}.json', parsed)
    run.state['reviews'].append(parsed)
    run.state['request_total'] = run.state.get('request_total', 0) + 1
    if not parsed['issues'] or r == REPAIRS:
        run.state['phase'] = 'complete'
        run.save_state()
        run.status(status='complete', repair_rounds=r, final_reviewer=parsed,
                   part_count=len(run.state['graph']['parts']), requests=run.state['request_total'],
                   request_breakdown=run.state['requests'])
        print(json.dumps({'round': r, 'coarse': 'complete',
                          'next': 'coarse stage finished; GPU worker will refine'}, indent=2))
        return
    run.state['round'] = r + 1
    run.state['phase'] = 'repair'
    run.save_state()
    print(json.dumps({'round': r, 'issues': len(parsed['issues']),
                      'next': 'prompt --stage builder --repair; then edit/done --repair'}, indent=2))


def cmd_fail(a):
    run = Run(a.run)
    run.state['phase'] = 'failed'
    run.save_state()
    run.status(status='failed', traceback=a.reason)
    print('marked failed')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('init')
    s.add_argument('--image', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--model', default='claude-opus-5-5')
    s.add_argument('--reasoning-effort', default='high', choices=['low', 'medium', 'high'])
    s.add_argument('--agent-identity', default='Claude Code subagent')
    s = sub.add_parser('prompt')
    s.add_argument('--stage', required=True, choices=['planner', 'builder', 'reviewer'])
    s.add_argument('--run')
    s.add_argument('--part')
    s.add_argument('--repair', action='store_true')
    for name in ('plan', 'edit', 'inspect', 'done', 'snapshot', 'review', 'fail'):
        s = sub.add_parser(name)
        s.add_argument('--run', required=True)
        if name in ('edit', 'inspect', 'done'):
            group = s.add_mutually_exclusive_group(required=True)
            group.add_argument('--part')
            group.add_argument('--repair', action='store_true')
        if name == 'plan':
            s.add_argument('--graph', required=True)
        if name == 'edit':
            s.add_argument('--edits', required=True)
        if name == 'done':
            s.add_argument('--text', required=True)
        if name == 'review':
            s.add_argument('--json', required=True)
        if name == 'fail':
            s.add_argument('--reason', required=True)
    a = p.parse_args(argv)
    if a.command == 'prompt' and a.stage == 'builder' and (not a.run or (not a.part and not a.repair)):
        p.error('builder prompt needs --run and --part or --repair')
    if a.command == 'edit':
        return cmd_edit(a, 'apply_voxel_edits')
    globals()['cmd_' + a.command](a)


if __name__ == '__main__':
    main()
