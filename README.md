<div align="center">

# PartBloom

**Structure First. Details Bloom.**

</div>

PartBloom is an open-source framework that uses multimodal LLMs to construct part-level voxel proxies, then transforms them into detailed 3D assets through generative diffusion models.

The current release is the **TRELLIS1 version**. A TRELLIS.2 version and automatic rigging are planned.

## Overview

Given a single image, PartBloom works in two stages:

1. **Structure first: a multimodal LLM agent builds a part-level voxel proxy.**
   - A *Planner* looks at the image and chooses the parts. It splits by function and independent motion, and keeps small fixed attachments with the part they belong to. There are no category templates, so it doesn't assume a fixed layout for chairs, cars and so on.
   - A *Builder* edits a shared 64³ voxel canvas directly with `polygon`, `brush`, `move` and `clear` tools, one part at a time.
   - A *Reviewer* compares renders of the proxy with the input image. It triggers up to two whole-object repair rounds and stops early if it finds nothing to fix.
2. **Details bloom: a generative diffusion model turns the proxy into a detailed 3D asset.**
   - The proxy surface is dilated into a thin support band and encoded by the TRELLIS sparse-structure VAE. It is then partially noised (t = 0.65) and refined by 12 image-guided denoising steps (CFG 5).
   - Part labels are transferred to the refined structure. One structured latent (SLat) is sampled for the whole object.
   - Each part is decoded from its own slice of the shared SLat. This yields one mesh per part that stays consistent with the rest of the object.
   - Textures come from TRELLIS's own Gaussian decoder and texture baking. The output is a textured GLB for every part plus an assembled GLB.

The agent stage is model-agnostic. It talks to the LLM through protocol adapters (`responses`, `generate_content`, `anthropic`, `chat_completions`), so switching to another provider only takes a config change. Results below use Gemini-4-Argon (medium), GPT-6-Astra (medium) and Claude Opus 5.5 (high).

## Roadmap

| Phase | Milestone | Deliverables | Status |
| --- | --- | --- | --- |
| 🚀 M0 | Project launch | This repository and qualitative comparisons | ✅ Done |
| 🧩 M1 | TRELLIS1 version | MLLM part planning, voxel proxy tools, image-guided structure refinement, shared-SLat per-part decoding, and native texture export | ✅ Done |
| 📦 M2 | Code release | Pipeline code, prompts, configs and reproduction scripts | 🚧 In progress |
| 🌸 M3 | TRELLIS.2 version | Port the detail-blooming stage to the TRELLIS.2 backbone | 🔜 Planned |
| 🦴 M4 | Rigging | Rigging demos on three assets (done); automatic skeletons, skinning and animation export for part-level assets (planned) | 🚧 Partial |

> **Status legend:** ✅ Done · 🚧 Partial or in progress · 🔜 Planned

## Results

The comparison has 14 single-image inputs: vehicles, mecha, creatures, statues, furniture, architecture and dioramas. Each cell shows that method's latest finished run, rendered by the method's own renderer with the same lights and framing on a white background, and turned to roughly match the input view.

### Image-to-3D comparison

| Method | Description |
| --- | --- |
| PhysX-Omni | Official fine-tuned VLM predicts voxels, which are decoded by TRELLIS (seed 1) |
| Direct Blender | An agent looks at the image and writes Blender Python directly: Argon (Antigravity CLI), Astra (Codex), Opus 5.5 (Claude Code) |
| **PartBloom** | Ours, with Gemini-4-Argon (medium), GPT-6-Astra (medium) and Claude Opus 5.5 (high) |

![image-to-3D comparison 1](assets/comparison_01.jpg)
![image-to-3D comparison 2](assets/comparison_02.jpg)
![image-to-3D comparison 3](assets/comparison_03.jpg)

### Part decomposition

Colors mark parts. SegviGen and PartLLM (general granularity) run with their official settings (seed 42). PartBloom parts come from the same runs shown in the image-to-3D comparison. All part cells use one shared renderer and palette, and translucent glass for the bottle so its interior stays visible.

![part comparison 1](assets/parts_01.jpg)
![part comparison 2](assets/parts_02.jpg)
![part comparison 3](assets/parts_03.jpg)

### Rigging

PartBloom outputs one mesh per functional, independently moving part, so its assets can be rigged directly. Each part is bound to a bone without re-generating or re-segmenting the model. The demos below were rigged and animated in Blender on top of the PartBloom part meshes and textures. Click a GIF for the full video.

| Covered Wagon: drive and steer | Mech Boar: run cycle | Mecha: shield charge |
| :---: | :---: | :---: |
| [![Covered Wagon](assets/rigging/covered_wagon.gif)](assets/rigging/covered_wagon.mp4) | [![Mech Boar](assets/rigging/mech_boar.gif)](assets/rigging/mech_boar.mp4) | [![Mecha](assets/rigging/mecha.gif)](assets/rigging/mecha.mp4) |
| PartBloom + Claude Opus 5.5 | PartBloom + GPT-6-Astra | PartBloom + Claude Opus 5.5 |

These are per-asset demos rather than an automatic rigging pipeline, which is still planned (M4).

## Coordinate convention

Voxel proxies use +X right, −Y forward and +Z up, on a 0..63 grid. Exported GLB files follow the standard Y-up convention.

## Acknowledgements

PartBloom builds on [TRELLIS](https://github.com/microsoft/TRELLIS). Baseline results use [PhysX-Omni](https://github.com/ziangcao0312/PhysX-Omni), SegviGen and PartLLM with their official settings.
