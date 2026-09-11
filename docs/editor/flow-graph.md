# Flow Graph

Flow is ARC's visual gameplay scripting system. A Flow Graph is authored as a `.arcflow` asset and uses the same shared graph-editor foundation as Material Graphs while keeping gameplay semantics in the Flow domain.

## F1 authoring contract

F1 defines the editable source representation only. Flow Graphs are **not** interpreted directly at runtime.

A Flow asset contains:

- schema version and asset name
- graph-local variable definitions
- typed nodes and stable node IDs
- typed connections and stable connection IDs
- editor viewport state

Execution and data are separate connection classes. Execution pins can connect only to execution pins. Value pins carry a declared value type and can connect only when their types are compatible.

Initial value types are `bool`, `int`, `float`, `vec2`, `vec3`, `vec4`, `string`, `name`, `entity`, and `component`. The internal `any` type is reserved for generic nodes.

Initial nodes are:

- Begin Play
- End Play
- Tick
- Fixed Tick
- Input Action
- Branch

Graph variables describe per-instance gameplay state. Variables can be marked `exposed`; later entity integration will surface those values as instance overrides in the inspector.

## Editor integration

Flow Graphs are created from the Content Browser and open in the Flow Graph Editor. The editor reuses ARC's shared viewport, pin, wire, selection, pan/zoom, and node interaction primitives. Flow-specific node definitions and connection rules are provided by `flowGraphDomain`.

The native editor host discovers `.arcflow` files for Content Browser persistence, but F1 intentionally does not add an engine runtime asset type or ECS execution path.

## Runtime boundary

The editable graph remains source authoring data:

```text
.arcflow
   |
   v
Flow validation / compiler     (F2)
   |
   v
Typed Flow IR / bytecode       (F2)
   |
   v
Flow VM                        (F3)
   |
   v
Stable project world API       (M3.5)
   |
   v
ECS / runtime world
```

F2 owns semantic validation, typed IR, diagnostics tied to node/pin IDs, and bytecode generation. F3 owns execution, per-instance VM state, lifecycle events, instruction budgets, and integration through the stable M3.5 world API rather than direct `ecs::world` access.
