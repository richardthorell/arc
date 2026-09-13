# Flow Play integration

Flow F6.1 binds authored `.arcflow` graphs to scene entities through the `Flow` component. The component stores a normalized path relative to the project Content directory plus an enabled flag.

When Play starts, ARC scans the isolated Play World for enabled Flow bindings. Each unique graph is compiled once into immutable bytecode while every bound entity owns an independent `vm_instance`, so variables and value slots do not leak between entities sharing one graph.

`Begin Play` is dispatched before the runtime world starts. Physical input and `Fixed Tick` run in the gameplay-command phase, frame `Tick` runs in presentation extraction, and `End Play` runs when the Play runtime world releases its Flow systems. Flow world operations are routed through the stable M3.5 `game_world_api_v1` bridge and retain scheduler command-buffer boundaries.

F6.1 maps the current physical Play input stream to deterministic Flow action tokens: `Key.<code>`, `MouseButton.<code>`, `MouseWheel`, and `Focus`. Semantic action-map names are a later input/Flow integration step.

Bindings are instantiated from authored scene state when Play starts. Runtime-created entities do not automatically gain a VM when a Flow component is attached during that session; hot lifecycle management remains a later Flow milestone.
