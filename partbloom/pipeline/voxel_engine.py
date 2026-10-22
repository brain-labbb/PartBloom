"""Agent-editable, part-aware 64^3 voxel canvas.

Internal coordinate convention: (x, y, z) with x right, y depth, z up.
The exported sparse array matches PhysX-Omni's ind_*.npy/allind.npy
convention: integer (N, 3) xyz, no batch index in the saved files.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, generate_binary_structure


class VoxelCanvas:
    def __init__(self, resolution: int = 64):
        if resolution != 64:
            raise ValueError("This MVP is intentionally locked to resolution=64")
        self.n = resolution
        self.parts: dict[str, np.ndarray] = {}
        self.action_log: list[dict[str, Any]] = []

    def _part(self, name: str) -> np.ndarray:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][\w-]{0,39}", name):
            raise ValueError("part must be a short name, like 'seat' or 'back_left_leg'")
        if name not in self.parts:
            self.parts[name] = np.zeros((self.n, self.n, self.n), dtype=bool)
        return self.parts[name]

    @staticmethod
    def _integer(value: Any, label: str, low: int, high: int) -> int:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{label} must be an integer")
        i = int(value)
        if i < low or i > high:
            raise ValueError(f"{label} must be between {low} and {high}")
        return i

    def paint_polygon(self, part: str, plane: str, start: int, end: int,
                      vertices: list[list[int]], mode: str = "add") -> dict:
        """Paint an arbitrary polygon on a 2D voxel slice range, inclusive.

        xy extrudes along z; xz along y; yz along x.
        This is editing the grid itself, not creating a persistent primitive.
        """
        if plane not in {"xy", "xz", "yz"}:
            raise ValueError("plane must be xy, xz or yz")
        if mode not in {"add", "erase"}:
            raise ValueError("mode must be add or erase")
        start = self._integer(start, "start", 0, self.n - 1)
        end = self._integer(end, "end", start, self.n - 1)
        if not 3 <= len(vertices) <= 24:
            raise ValueError("polygon needs 3..24 vertices")
        clean = []
        for v in vertices:
            if not isinstance(v, (tuple, list)) or len(v) != 2:
                raise ValueError("each vertex must be [u,v]")
            clean.append(tuple(self._integer(c, "vertex coordinate", 0, self.n - 1) for c in v))
        im = Image.new("1", (self.n, self.n), color=0)
        ImageDraw.Draw(im).polygon(clean, fill=1)
        mask_uv = np.asarray(im, dtype=bool).T  # PIL is [v,u], grid is [u,v]
        if not mask_uv.any():
            raise ValueError("polygon has zero area")
        arr = self._part(part)
        change = arr if mode == "add" else ~arr
        before = int(arr.sum())
        for k in range(start, end + 1):
            if plane == "xy":
                plane_data = arr[:, :, k]
            elif plane == "xz":
                plane_data = arr[:, k, :]
            else:
                plane_data = arr[k, :, :]
            if mode == "add":
                plane_data |= mask_uv
            else:
                plane_data &= ~mask_uv
        return {"part": part, "operation": "polygon", "voxel_count": int(arr.sum()),
                "delta": int(arr.sum()) - before}

    def paint_brush(self, part: str, points: list[list[int]], radius: int = 2,
                    mode: str = "add") -> dict:
        """Paint a 3D polyline with a spherical voxel brush."""
        radius = self._integer(radius, "radius", 1, 8)
        if mode not in {"add", "erase"}:
            raise ValueError("mode must be add or erase")
        if not isinstance(points, list) or not 1 <= len(points) <= 32:
            raise ValueError("points must contain 1..32 [x,y,z] locations")
        points_arr = []
        for p in points:
            if not isinstance(p, (tuple, list)) or len(p) != 3:
                raise ValueError("each point must be [x,y,z]")
            points_arr.append(np.array([self._integer(v, "brush coordinate", 0, self.n-1) for v in p]))
        arr = self._part(part)
        before = int(arr.sum())
        centers = []
        for a, b in zip(points_arr, points_arr[1:]):
            steps = int(np.max(np.abs(b - a))) + 1
            centers.extend(np.rint(np.linspace(a, b, steps)).astype(int))
        if len(points_arr) == 1:
            centers = [points_arr[0]]
        for p in centers:
            x0, y0, z0 = np.maximum(p - radius, 0)
            x1, y1, z1 = np.minimum(p + radius + 1, self.n)
            x, y, z = np.ogrid[x0:x1, y0:y1, z0:z1]
            brush = (x-p[0])**2 + (y-p[1])**2 + (z-p[2])**2 <= radius**2
            region = arr[x0:x1, y0:y1, z0:z1]
            if mode == "add":
                region |= brush
            else:
                region &= ~brush
        return {"part": part, "operation": "brush", "voxel_count": int(arr.sum()),
                "delta": int(arr.sum()) - before}

    def translate(self, part: str, delta: list[int]) -> dict:
        if part not in self.parts:
            raise ValueError(f"unknown part: {part}")
        if not isinstance(delta, (tuple, list)) or len(delta) != 3:
            raise ValueError("delta must be [dx,dy,dz]")
        delta = np.array([self._integer(v, "delta", -63, 63) for v in delta])
        src = np.argwhere(self.parts[part])
        shifted = src + delta
        dst = np.zeros_like(self.parts[part])
        mask = np.all((shifted >= 0) & (shifted < self.n), axis=1)
        valid = shifted[mask]
        if len(valid):
            dst[tuple(valid.T)] = True
        self.parts[part] = dst
        return {"part": part, "operation": "translate", "kept": int(mask.sum()),
                "clipped": int((~mask).sum())}

    def clear(self, part: str) -> dict:
        if part not in self.parts:
            raise ValueError(f"unknown part: {part}")
        count = int(self.parts.pop(part).sum())
        return {"part": part, "operation": "clear", "removed": count}

    def apply(self, edit: dict) -> dict:
        """Validated executor for LLM-suggested edit operations; never eval code."""
        op = edit.get("op")
        if op == "polygon":
            result = self.paint_polygon(edit["part"], edit["plane"], edit["start"],
                                        edit["end"], edit["vertices"], edit.get("mode", "add"))
        elif op == "brush":
            result = self.paint_brush(edit["part"], edit["points"],
                                      edit.get("radius", 2), edit.get("mode", "add"))
        elif op == "move":
            result = self.translate(edit["part"], edit["delta"])
        elif op == "clear":
            result = self.clear(edit["part"])
        else:
            raise ValueError(f"unknown edit op: {op}")
        self.action_log.append({"edit": edit, "result": result})
        return result

    def summary(self) -> dict:
        parts = {}
        union = np.zeros((self.n,)*3, dtype=bool)
        for name, vol in self.parts.items():
            coords = np.argwhere(vol)
            overlap = int((union & vol).sum())
            union |= vol
            parts[name] = {
                "occupied": int(len(coords)),
                "bbox_min": coords.min(axis=0).tolist() if len(coords) else None,
                "bbox_max": coords.max(axis=0).tolist() if len(coords) else None,
                "overlap_with_earlier_parts": overlap,
            }
        return {"resolution": self.n, "coordinate_order": "xyz", "parts": parts,
                "total_unique_solid_voxels": int(union.sum()),
                "edit_count": len(self.action_log)}

    def export(self, output_dir: str | Path, reference_image: str | Path | None = None) -> dict:
        """Save dense parts + sparse surface coords for PhysX-Omni.

        The PhysX-Omni decoder reads ind_{i}.npy concatenated in ascending i,
        with allind.npy = concatenation of the same coordinates (not [b,x,y,z]).
        The paper's RLE is not required for this downstream interface.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if reference_image is not None:
            Image.open(reference_image).convert("RGB").save(out / "cond_img.png")
        for old in out.glob("ind_*.npy"):
            old.unlink()
        names = list(self.parts)
        volumes = np.stack([self.parts[name] for name in names], axis=0) if names else np.zeros((0,64,64,64), bool)
        np.savez_compressed(out / "dense_parts.npz", names=np.array(names), volumes=volumes)
        surface_st = generate_binary_structure(3, 1)  # 6-neighborhood
        used = np.zeros((self.n,)*3, bool)
        sparse = []
        index = []
        for name, vol in self.parts.items():
            surf = vol & ~binary_erosion(vol, structure=surface_st, border_value=0)
            unique = surf & ~used
            used |= surf
            coords = np.argwhere(unique).astype(np.int64)
            if not len(coords):
                continue
            np.save(out / f"ind_{len(index)}.npy", coords)
            sparse.append(coords)
            index.append({"index": len(index), "part": name, "coords": int(len(coords)),
                          "discarded_duplicate_surface_coords": int(surf.sum()-unique.sum())})
        if not sparse:
            raise ValueError("Cannot export an empty voxel canvas")
        concat = np.concatenate(sparse, axis=0)
        np.save(out / "allind.npy", concat)
        (out / "part_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf8")
        (out / "summary.json").write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf8")
        (out / "edits.json").write_text(json.dumps(self.action_log, ensure_ascii=False, indent=2), encoding="utf8")
        return {"parts": len(index), "sparse_coords": int(len(concat)), "output_dir": str(out)}

    @classmethod
    def from_npz(cls, path: str | Path) -> "VoxelCanvas":
        with np.load(path) as data:
            names = [str(n) for n in data["names"]]
            volumes = data["volumes"]
        canvas = cls(64)
        for name, volume in zip(names, volumes):
            if volume.shape != (64,64,64):
                raise ValueError("Bad volume shape")
            canvas.parts[name] = volume.astype(bool)
        return canvas
