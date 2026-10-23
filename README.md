<div align="center">

# PartBloom

**Structure First. Details Bloom.**

</div>

PartBloom is an open-source agentic framework that uses LLMs (Gemini-4-Argon, GPT-6-Astra, Claude Opus 5.5) to construct part-level coarse proxies, then transforms them into detailed 3D assets through generative diffusion models. By combining the planning and structure-construction strengths of LLMs with the high-fidelity detail generation of diffusion models, PartBloom points to a possible future for 3D generation.

The current release is the **TRELLIS version**.

## Overview

Given a single image, PartBloom works in two stages:

1. **Structure first: an LLM agent builds a part-level coarse proxy.**
   - A *Planner* looks at the image and decides the parts, splitting by function and independent motion.
   - A *Builder* edits a shared 64³ voxel canvas directly with `polygon`, `brush`, `move` and `clear` tools, one part at a time.
   - A *Reviewer* compares renders of the proxy with the input and triggers up to two repair rounds.
2. **Details bloom: a diffusion model turns the proxy into a detailed 3D asset.**
   - The proxy is encoded by the TRELLIS sparse-structure VAE, partially noised, and refined with image guidance.
   - Part labels are transferred to the refined structure and one shared structured latent is sampled for the whole object.
   - Each part is decoded from its own slice of the latent, so parts come out consistent with each other. Textures come from TRELLIS's own Gaussian decoder and texture baking.

The agent stage is model-agnostic — switching providers is a config change. Results below use Gemini-4-Argon (medium), GPT-6-Astra (medium) and Claude Opus 5.5 (high).

## Roadmap

| Phase | Milestone | Deliverables | Status |
| --- | --- | --- | --- |
| 🚀 M0 | Project launch | This repository and qualitative comparisons | ✅ Done |
| 🧩 M1 | TRELLIS version | LLM part planning, coarse-proxy building tools, image-guided structure refinement, shared-SLat per-part decoding, and native texture export | ✅ Done |
| 📦 M2 | Code release | Pipeline code, prompts and configs | ✅ Done |
| 🦴 M3 | Rigging | Skeletons, skinning and animation for part-level assets | 🚧 In progress |
| 🌸 M4 | TRELLIS.2 version | Port the detail-blooming stage to the TRELLIS.2 backbone | 🔜 Planned |

> **Status legend:** ✅ Done · 🚧 Partial or in progress · 🔜 Planned

## Results

The benchmark has 14 single-image inputs: vehicles, mecha, creatures, statues, furniture, architecture and dioramas.

### Image-to-3D comparison

| Method | Description |
| --- | --- |
| PhysX-Omni | Official fine-tuned VLM directly predicts a voxel structure, which is decoded by TRELLIS (seed 1) |
| Direct Blender | An agent looks at the image and writes Blender Python directly: Gemini-4-Argon (Antigravity CLI), GPT-6-Astra (Codex), Claude Opus 5.5 (Claude Code) |
| **PartBloom** | Ours, with Gemini-4-Argon (medium), GPT-6-Astra (medium) and Claude Opus 5.5 (high) |

![image-to-3D comparison 1](assets/comparison_01.jpg)
![image-to-3D comparison 2](assets/comparison_02.jpg)
![image-to-3D comparison 3](assets/comparison_03.jpg)

### Part decomposition

PartBloom generates part-level assets natively, so we also compare its parts with dedicated part segmentation methods, SegviGen and PartLLM, which split a 3D asset into parts. Colors mark parts. PartLLM uses its general granularity.

![part comparison 1](assets/parts_01.jpg)
![part comparison 2](assets/parts_02.jpg)
![part comparison 3](assets/parts_03.jpg)

### Rigging

PartBloom outputs one mesh per functional, independently moving part, so its assets can be rigged directly. Click a GIF for the full video.

| Covered Wagon: drive and steer | Mech Boar: run cycle | Mecha: shield charge |
| :---: | :---: | :---: |
| [![Covered Wagon](assets/rigging/covered_wagon.gif)](assets/rigging/covered_wagon.mp4) | [![Mech Boar](assets/rigging/mech_boar.gif)](assets/rigging/mech_boar.mp4) | [![Mecha](assets/rigging/mecha.gif)](assets/rigging/mecha.mp4) |
| PartBloom + Claude Opus 5.5 | PartBloom + GPT-6-Astra | PartBloom + Claude Opus 5.5 |

## Getting started

### Setup

1. Clone [PhysX-Omni](https://github.com/ziangcao0312/PhysX-Omni) → `vendor/PhysX-Omni` and [DINOv2](https://github.com/facebookresearch/dinov2) → `vendor/dinov2`.
2. Download [TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) → `weights/TRELLIS-image-large` and `dinov2_vitl14_reg4_pretrain.pth` → `weights/dinov2/`.
3. Create a GPU environment at `.venv/` with CUDA, xatlas, nvdiffrast and the Gaussian renderer.
4. Install [Blender](https://www.blender.org/) 4.5 and put `blender` on `PATH`.
5. Install CPU dependencies and verify:

```bash
pip install -r partbloom/requirements-cpu.txt
python partbloom/run.py doctor
```

### API config

Copy `partbloom/configs/api.example.json` and fill in your provider:

```json
{
  "protocol": "responses",
  "base_url": "https://your-service.example/v1",
  "api_key_env": "PARTBLOOM_API_KEY"
}
```

`protocol` is one of `responses`, `generate_content`, `anthropic` or `chat_completions`.

### Run

```bash
python partbloom/run.py run \
  --image path/to/image.png \
  --api-config my_api.json --model claude-opus-5-5 --reasoning-effort high \
  --out outputs/my_asset --gpu 0
```

A full run writes `coarse/`, `refined/`, `textured/` (per-part GLBs and `assembly_textured.glb`) and `parts.png`. Add `--resume` to skip completed stages. Run `python partbloom/run.py --help` for individual stage commands.

## Acknowledgements

PartBloom builds on [TRELLIS](https://github.com/microsoft/TRELLIS). Baselines include [PhysX-Omni](https://github.com/ziangcao0312/PhysX-Omni), Direct Blender agents, SegviGen and PartLLM.

## License

PartBloom is released under the [MIT License](LICENSE). Its dependencies keep their own licenses: TRELLIS is MIT, and PhysX-Omni, whose TRELLIS implementation PartBloom runs, uses the S-Lab License 1.0, which permits non-commercial use only; commercial use requires contacting its authors.
