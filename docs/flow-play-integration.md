# Flow Play integration

Flow F6.1 binds authored `.arcflow` graphs to scene entities through the `Flow` component. The component stores a normalized path relative to the project Content directory plus an enabled flag. A newly added Flow component with no graph selected is inert, so entities can be authored incrementally without blocking Play startup.

When Play starts, ARC scans the isolated Play World for enabled Flow bindings. Each unique graph is compiled once into immutable bytecode while every bound entity owns an independent `vm_instance`, so variables and value slots do not leak between entities sharing one graph.

F6.2 keeps that binding set live for the rest of the Play session. ARC consumes ECS structural/component change journals for `Flow`, `Active`, and destroyed entities instead of rescanning the whole world each frame. Lifecycle reconciliation runs before gameplay and again after the fixed phases so changes that already exist at the start of a tick can receive `Begin Play` before `Fixed Tick`, while bindings created or changed during fixed simulation are reconciled before frame `Tick`.

A runtime entity gains a Flow VM when it becomes alive, active, enabled, and has a non-empty valid graph path. Removing or disabling its Flow component, making the entity inactive, changing the graph path, or destroying the entity retires the old VM. Live entities receive `End Play` before the VM is removed; destroyed entities are retired without turning an already-valid destruction into a world fault. Re-enabling a binding or assigning a new graph creates fresh per-entity VM state and dispatches `Begin Play` again.

Physical input and `Fixed Tick` run in the gameplay-command phase, frame `Tick` runs in presentation extraction, and Flow world operations remain routed through the stable M3.5 `game_world_api_v1` bridge with scheduler command-buffer boundaries. Graph bytecode is cached by Content-relative path for the Play session, while VM state remains per entity.

## Semantic input during Play

Flow `Input Action` nodes consume project-authored semantic action names from `Config/Input.json`. The Play runtime loads that file when the Flow Play session is installed, applies its contexts and bindings to logical player `0`, and collects the declared action names for Flow dispatch. A graph therefore refers to an action such as `Jump`, while the project config decides which physical control activates it. Remapping `Jump` from `Space` to `Enter`, for example, changes the runtime behavior without changing the graph.

During each fixed gameplay step ARC samples the Play input snapshot into the input system, evaluates the configured actions, and dispatches Flow input events for state transitions. A newly pressed action invokes the graph's `triggered` output with value `1.0`; a released action invokes `completed` with value `0.0`. This is the path covered by `flow_semantic_input_tests.cpp`.

Current `Config/Input.json` parsing supports keyboard keys and mouse buttons, including the existing binding processors (`scale`, `invert`, and `clamp`). The underlying Play input sampler also receives mouse position, mouse wheel, and focus changes, but those axis/wheel controls are not yet representable as project action bindings. Gamepad controls, touch, gyro/accelerometer, and composite/vector bindings are likewise future input-config work.

Flow currently evaluates semantic actions only for logical player `0`; per-entity/player routing is not implemented yet. If the default project `Config/Input.json` is absent, Play can still start, but there are no configured semantic actions to dispatch. An explicitly supplied input-config path must exist and parse successfully. Input mappings are loaded when the Play session is created, so editing `Config/Input.json` during an active session does not hot-reload the mapping yet.

Editing the source contents of an already-cached `.arcflow` file in place is also not a hot-reload signal yet; runtime bytecode asset reload/versioning remains a later milestone.
