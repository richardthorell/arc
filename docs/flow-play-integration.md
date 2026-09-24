# Flow Play integration

Flow F6.1 binds authored `.arcflow` graphs to scene entities through the `Flow` component. The component stores a normalized path relative to the project Content directory plus an enabled flag. A newly added Flow component with no graph selected is inert, so entities can be authored incrementally without blocking Play startup.

When Play starts, ARC scans the isolated Play World for enabled Flow bindings. Each unique graph is compiled into an immutable bytecode artifact while every bound entity owns an independent `vm_instance`, so variables and value slots do not leak between entities sharing one graph.

F6.2 keeps that binding set live for the rest of the Play session. ARC consumes ECS structural/component change journals for `Flow`, `Active`, and destroyed entities instead of rescanning the whole world each frame. Lifecycle reconciliation runs before gameplay and again after the fixed phases so changes that already exist at the start of a tick can receive `Begin Play` before `Fixed Tick`, while bindings created or changed during fixed simulation are reconciled before frame `Tick`.

A runtime entity gains a Flow VM when it becomes alive, active, enabled, and has a non-empty valid graph path. Removing or disabling its Flow component, making the entity inactive, changing the graph path, or destroying the entity retires the old VM. Live entities receive `End Play` before the VM is removed; destroyed entities are retired without turning an already-valid destruction into a world fault. Re-enabling a binding or assigning a new graph creates fresh per-entity VM state and dispatches `Begin Play` again.

Physical input and `Fixed Tick` run in the gameplay-command phase, frame `Tick` runs in presentation extraction, and Flow world operations remain routed through the stable M3.5 `game_world_api_v1` bridge with scheduler command-buffer boundaries. Compiled artifacts are cached by Content-relative path for the Play session, while VM state remains per entity.

## Flow artifact generations and Play hot reload

Each cached Flow program has a Play-session generation. Generation `1` is the first successfully compiled source for that Content-relative graph path. ARC checks the source used by active/cached Flow programs during fixed gameplay; when the source changes and compiles successfully, ARC publishes a new immutable bytecode generation for that path and moves every bound instance still using the previous generation onto it.

Hot reload deliberately uses **restart semantics rather than state migration**. Before switching generations, each live old VM receives `End Play`. ARC then creates a fresh VM from the new bytecode defaults and dispatches `Begin Play`. Graph variables, value slots, latent actions, and other per-instance runtime state are therefore reset on reload. Entity/world state changed through normal Flow world operations remains ordinary world state and is not copied through a special Flow migration path.

A changed source is never published unless it compiles into a valid VM program. If the file cannot be read, compilation fails, or the replacement VM is invalid, ARC keeps the last-good generation and its existing bound VMs running. The failure is reported through the Flow diagnostics log, and an unchanged rejected source is not repeatedly recompiled every fixed step. Changing the source again retries the reload, so correcting an invalid edit can recover the Play session without restarting it.

## Semantic input during Play

Flow `Input Action` nodes consume project-authored semantic action names from `Config/Input.json`. The Play runtime loads that file when the Flow Play session is installed, applies its contexts and bindings to logical player `0`, and collects the declared action names for Flow dispatch. A graph therefore refers to an action such as `Jump`, while the project config decides which physical control activates it. Remapping `Jump` from `Space` to `Enter`, for example, changes the runtime behavior without changing the graph.

During each fixed gameplay step ARC samples the Play input snapshot into the input system, evaluates the configured actions, and dispatches Flow input events for state transitions. A newly pressed action invokes the graph's `triggered` output with value `1.0`; a released action invokes `completed` with value `0.0`. This is the path covered by `flow_semantic_input_tests.cpp`.

Current `Config/Input.json` parsing supports keyboard keys and mouse buttons, including the existing binding processors (`scale`, `invert`, and `clamp`). The underlying Play input sampler also receives mouse position, mouse wheel, and focus changes, but those axis/wheel controls are not yet representable as project action bindings. Gamepad controls, touch, gyro/accelerometer, and composite/vector bindings are likewise future input-config work.

Flow currently evaluates semantic actions only for logical player `0`; per-entity/player routing is not implemented yet. If the default project `Config/Input.json` is absent, Play can still start, but there are no configured semantic actions to dispatch. An explicitly supplied input-config path must exist and parse successfully. Input mappings are loaded when the Play session is created, so editing `Config/Input.json` during an active session does not hot-reload the mapping yet.
