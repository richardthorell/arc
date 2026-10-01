# Animation runtime foundation

ARC animation is an engine-owned runtime subsystem. Importers, scene/ECS code, renderer backends, and editor tools consume the contracts in `arc/animation/animation.h`; none of those layers own the evaluator itself.

## M1 scope

M1 establishes the executable runtime and data-model foundation only. It deliberately does not choose a production compressed clip format, a skinning backend, an animation graph format, or an editor authoring workflow.

The runtime owns registered skeleton/clip resources and animation instances. Skeletons use deterministic parent-before-child joint ordering. Clips use sparse translation, rotation, and scale channels keyed in seconds. Missing channels inherit the skeleton bind pose.

## Transform conventions

- A `joint_transform` is translation + quaternion rotation + component-wise scale.
- Local transforms are relative to the joint parent.
- Model transforms are derived in deterministic joint order from the hierarchy.
- Translation uses the same units as ARC scene transforms.
- Rotation quaternions are stored `(x, y, z, w)` and must be finite/non-zero.
- Animation does not perform source-coordinate conversion. Import/cook code is responsible for converting incoming data to ARC's scene/math convention.
- Parent composition applies parent scale to the child translation, rotates it by the parent rotation, then applies parent translation. Rotation composes parent then child.
- Optional inverse-bind matrices are animation-owned data. They are handed to rendering as plain ARC math values; renderer buffer/resource types stay outside the animation API.

## Time and sampling

Clip/key times are seconds. M1 clips are sparse and uncompressed so the evaluator and ownership rules can be tested before production clip cooking/compression lands.

For non-looping clips, sample time clamps to `[0, duration]`. For looping clips, time wraps modulo duration, including negative explicit sample times. Translation and scale interpolate linearly. Rotation uses shortest-path normalized quaternion interpolation. A channel with no keys preserves the bind-pose value for that channel.

Later cooked/compressed clips may replace the storage representation, but they must preserve these public timing and complete-pose semantics unless the animation API is versioned deliberately.

## Runtime ownership and publication

`animation_runtime` owns three independent runtime handle spaces:

1. registered skeleton resources,
2. registered clip resources,
3. animation instances.

A clip is registered against exactly one skeleton. An instance references one skeleton and optionally one compatible clip. Resources cannot be removed while still referenced by dependent runtime objects.

Each instance owns two complete pose buffers. Evaluation writes only to the non-published buffer and atomically publishes it after local and model transforms are complete. This prevents consumers from observing a partially evaluated pose during one publication.

Lifecycle/resource mutation is currently a simulation-thread responsibility. Published pose reads may be consumed by downstream systems during the frame. A consumer must finish reading before two subsequent publications can recycle the same buffer. This makes the current contract compatible with moving pose evaluation onto ARC jobs later without exposing job-system primitives in the public animation API.

## Integration seams

### Assets/import

M1 accepts ARC-owned `skeleton_definition` and `animation_clip_definition` values. Future FBX/glTF importers and the cooker should translate source data into versioned ARC animation assets, then register/load them through this same runtime boundary. Source SDK types must not enter `engine/animation` public headers.

### Scene/ECS

Future Animator/Skeleton components should persist asset references and authoring/runtime-control state only. ECS storage must not contain evaluator internals or renderer skinning resources. Scene synchronization creates/configures/destroys `animation_runtime` instances explicitly.

### Renderer/skinning

`skinning_pose_view` exposes complete model transforms, optional inverse-bind matrices, and a pose generation. A later skinning implementation may turn those values into matrices/buffers appropriate for CPU, compute, vertex, or mesh-shader skinning. The animation module never exposes those renderer/backend objects.

### Animation Graph

The future Animation Graph is a control layer over this evaluator. Graph nodes/state machines should select clips, parameters, blends, masks, and procedural operations that evaluate into the same `animation_pose` contract; they must not introduce a parallel animation runtime.

## Follow-up milestones

M2+ can build on this foundation for imported/cooked skeleton and clip assets, production sampling/compression, skinning integration, blending/state machines, ECS authoring, Animation Graph tooling, asset preview, root motion/events, IK, retargeting, and diagnostics.
