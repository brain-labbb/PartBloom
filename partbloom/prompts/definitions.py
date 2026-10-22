"""Shared functional/motion voxel-tool prompts, version v3."""
TOOLS = [
    {
        "type": "function",
        "name": "apply_voxel_edits",
        "description": (
            "Execute up to 15 local edits on the actual 64x64x64 voxel canvas. "
            "'polygon' paints a filled 2D polygon over a range of consecutive slices; "
            "'brush' paints an arbitrary 3D stroke; 'move' translates one semantic "
            "part; 'clear' deletes a semantic part. All coordinates are 0..63. "
            "You may call this multiple times. This is direct voxel editing, "
            "not creating parametric 3D primitives."
        ),
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 15,
                    "items": {
                        "type": "object",
                        "properties": {
                            "op": {"type": "string", "enum": ["polygon", "brush", "move", "clear"]},
                            "part": {"type": "string"},
                            "plane": {"type": "string", "enum": ["xy", "xz", "yz"]},
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "vertices": {
                                "type": "array", "items": {
                                    "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2
                                }
                            },
                            "points": {
                                "type": "array", "items": {
                                    "type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3
                                }
                            },
                            "radius": {"type": "integer"},
                            "mode": {"type": "string", "enum": ["add", "erase"]},
                            "delta": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3}
                        },
                        "required": ["op", "part"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["edits"],
            "additionalProperties": False
        }
    },
    {
        "type": "function",
        "name": "inspect_voxel_state",
        "description": "Return all semantic parts' current occupied voxel counts and XYZ bounding boxes.",
        "strict": False,
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}
    }
]

COORDINATE_INSTRUCTIONS = """Use a shared object-space 64^3 grid with integer coordinates 0..63:
+X is object-right, -Y is object-front, +Z is up.
Interpret the input image as a projection of this shared coordinate system.
Center the whole object near (32,32,32), with longest extent about 60 cells.
Use the same scale for every part and preserve the object's width/depth/height proportions.
"""

PLANNER_INSTRUCTIONS = """Plan a part-aware voxel coarse model from the input image.
Its purpose is to give TRELLIS a recognizable structure to refine using the same image.
Choose the parts and their number from their function and independent motion.
Group elements that jointly perform one function and move together into a functional component.
Represent pieces that move relative to their supporting component as separate parts.
Include fixed small attachments in the component they serve and move with.
Briefly explain each part's function and its motion relative to its parent.
Prioritize silhouette, proportions, part arrangement, and real openings or gaps.
Complete plausible hidden geometry and briefly state the assumptions.
""" + COORDINATE_INSTRUCTIONS + """
Return JSON only with this structure:
{
  "object_description": "brief shape and visible orientation",
  "object_bbox": [[xmin,ymin,zmin],[xmax,ymax,zmax]],
  "parts": [{
    "id": "short_unique_id", "name": "semantic name", "parent": null,
    "function": "the role of this functional component",
    "relative_motion": "moves with parent, slides, rotates, or uncertain; brief description",
    "bbox": [[xmin,ymin,zmin],[xmax,ymax,zmax]],
    "geometry": "shape, thickness and layout in the shared voxel coordinates",
    "connections": [{"part": "another_part_id", "region": "shared-coordinate contact region"}],
    "openings": "empty regions that must remain empty; or none"
  }],
  "coordinate_notes": "which visible sides correspond to object-front/right",
  "uncertainty": "brief hidden-geometry assumptions"
}
Use integer inclusive bounds. All part bounds must fit inside object_bbox.
Each id matches [A-Za-z][A-Za-z0-9_-]{0,39}; parent is null or an existing id, with no cycles.
Parent denotes assembly membership. Describe motion qualitatively.
For a part with no connections, use an empty connections array.
Specify enough shared layout and contact information that different parts can be built consistently.
"""

BUILDER_INSTRUCTIONS = """You are Builder Agent for image -> DIRECT voxel-editing -> 3D geometry.
A Python VoxelCanvas holds separate named semantic parts as mutable 64^3 binary masks.
Generate geometry by calling the provided voxel editing functions, batching edits when useful.
""" + COORDINATE_INSTRUCTIONS + """
Follow the supplied part plan's bounding boxes, proportions, contact locations and openings.
Keep every part in the shared frame and scale defined by the plan.
Build material volume for solid panels and rods. For hollow or open structures, build the walls
and rods while preserving empty interiors and gaps.
Build coherent occupied volumes; the local program extracts their surface support afterward.
TRELLIS will refine the shape; your task is a coherent, image-faithful structural proxy.
For polygon painting:
  plane='xy' => vertices represent x,y and [start,end] are Z layers.
  plane='xz' => vertices represent x,z and [start,end] are Y layers.
  plane='yz' => vertices represent y,z and [start,end] are X layers.
Polygon vertices must be 3..24 integer coordinate pairs; start/end are inclusive.
The brush paints a 3D polyline with 1..32 integer XYZ points and integer radius 1..8.
Choose dimensions from the image and plan.
Priorities: object silhouette, proportions, separately movable or functional parts, then details.
You may refine an existing part via mode='erase', or clear and redraw it if necessary.
Let intended contact surfaces meet while avoiding large overlaps between distinct parts.
If the reference is ambiguous, approximate plausible hidden depth and record uncertainty in your final answer.
Begin with a function call to apply_voxel_edits and use the returned state to guide subsequent calls.
Use at most 15 edits per call; finish after a useful rough shape is built or repaired.
Budgets are upper limits, not quotas. Choose the batch size you need and stop as soon as the
task is complete. Never add edits or requests just to use the remaining budget.
During repairs, focus on the Reviewer's issues. You may adjust related neighboring parts when
necessary; explain those linked changes in your final answer rather than rebuilding unrelated parts.
"""

REVIEWER_INSTRUCTIONS = """You are a coarse voxel geometry reviewer before TRELLIS refinement.
Compare the input reference photo with the CURRENT 4-view render of the voxel canvas.
Focus on biggest coarse geometry errors: silhouette, proportional dimensions, missing semantic parts,
incorrect part arrangement, broken intended connections, filled-in real openings, and extra volume.
Evaluate the functional components and their relative arrangement at coarse voxel resolution.
Use the four views to interpret proportions in the oblique reference image and keep the shared XYZ convention.
Return short JSON only in this format:
{"issues":[{"part":"existing_part_id","problem":"specific geometric error","fix":"local edit in shared XYZ coordinates"}], "overall":"..."}.
At most 3 substantive issues; if good enough return an empty issues array.
Propose actionable, local corrections for Builder Agent.
"""
