import json

import pytest
from PIL import Image

from agent_runner import run_builder
from voxel_engine import VoxelCanvas


def message(text='done'):
    return {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]}


def paint(part):
    return {'output': [{'type': 'function_call', 'name': 'apply_voxel_edits', 'call_id': part,
                       'arguments': json.dumps({'edits': [{'op': 'brush', 'part': part,
                                                        'points': [[20, 20, 20]], 'radius': 1}]})}]}


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def call(self, inputs, *args):
        self.requests.append(list(inputs))
        return next(self.responses)


def test_prose_cannot_finish_an_empty_part(tmp_path):
    image = tmp_path/'input.png'; Image.new('RGB', (4, 4)).save(image)
    canvas = VoxelCanvas(); canvas._part('door')
    client = Client([message('Tool unavailable; here is JSON instead.'), paint('door'), message()])
    run_builder(client, 'test', canvas, image, tmp_path, 0, part_id='door', max_api_steps=3)
    assert canvas.parts['door'].any()
    assert 'empty_or_missing_parts' in json.dumps(client.requests[1])
    assert len(client.requests) == 3


def test_repair_cannot_finish_with_a_missing_planned_part(tmp_path):
    image = tmp_path/'input.png'; Image.new('RGB', (4, 4)).save(image)
    canvas = VoxelCanvas()
    client = Client([message(), paint('door'), message()])
    run_builder(client, 'test', canvas, image, tmp_path, 1,
                graph={'parts': [{'id': 'door'}]}, max_api_steps=3)
    assert canvas.parts['door'].any()


def test_invalid_completion_is_bounded_and_trace_saved(tmp_path):
    image = tmp_path/'input.png'; Image.new('RGB', (4, 4)).save(image)
    client = Client([message(), message()])
    with pytest.raises(RuntimeError, match='maximum API steps'):
        run_builder(client, 'test', VoxelCanvas(), image, tmp_path, 0,
                    part_id='door', max_api_steps=2)
    assert len(client.requests) == 2
    assert (tmp_path/'builder_round0_door.json').is_file()
