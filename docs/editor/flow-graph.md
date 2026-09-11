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

## F2 compiler contract

F2 adds the engine-side `arc-flow` module and treats `.arcflow` as compiler input rather than runtime data. `arc::flow::compile_asset` parses the authored JSON, performs semantic validation, lowers the graph to typed IR, and then lowers the IR to runtime-ready bytecode structures. Installed SDK packages export the module as `arc::Flow`.

Compiler diagnostics carry stable codes plus node, pin, and connection IDs where applicable. The initial validation pass rejects malformed schemas, duplicate IDs, unknown pins, incompatible value connections, multiple sources for one input, implicit execution fan-out, and execution cycles. Unreachable executable nodes are reported as warnings. Input Action nodes also require a non-empty action name.

The typed IR explicitly models:

- graph variables and typed defaults
- typed value slots
- lifecycle/input entry points
- event-value bindings such as delta time and input action values
- branch instructions and control-flow targets
- source node IDs for diagnostics and debugging

Bytecode removes editor-only graph topology while retaining the typed slots, entry points, compact instructions, and an instruction-to-node source map. `invalid_instruction` is the explicit return/end-of-flow target. Unconnected Branch conditions currently lower to the type default (`false`), keeping the bytecode deterministic until literal/value nodes expand the language.

Execution outputs intentionally allow only one target in F2. Future fan-out is represented by an explicit Sequence node so execution order stays deterministic rather than depending on connection storage order.

The F2 compiler does **not** execute bytecode and does not access ECS or the project world.

## F3 runtime contract

F3 adds `arc::flow::vm_instance`, the per-instance bytecode runtime. A VM references one immutable `bytecode_program`; the owning compiled Flow asset must outlive all VM instances that reference it. Runtime state is separate for every instance and contains graph-variable values, typed value slots, lifecycle state, and execution limits.

The VM dispatches Begin Play, End Play, Tick, Fixed Tick, and Input Action events. Tick and input event values are written through the entry-point bindings emitted by F2. Input Action dispatch matches the authored action name exactly. Tick, Fixed Tick, and input events execute only while the instance is active between Begin Play and End Play.

Each event dispatch has one aggregate instruction budget, defaulting to 4096 bytecode instructions. Exceeding the budget stops execution and reports the stopped instruction plus the source node when a source map is available. This guard remains mandatory even though F2 currently rejects authored execution cycles, because future loop/latent constructs and externally loaded bytecode must not be able to stall the runtime or editor.

VM construction validates bytecode versioning, variable/value defaults, entry-point targets and bindings, and instruction operands before execution. `reset()` restores graph-variable defaults and bytecode slot defaults and leaves the VM inactive. Exposed variable overrides can therefore be applied per instance before Begin Play without modifying the shared Flow asset.

F3 intentionally remains independent of ECS and the project world. F4 adds gameplay/world instructions and routes them through the stable M3.5 world API rather than exposing `ecs::world` to Flow.

## Runtime boundary

The editable graph remains source authoring data:

```text
.arcflow
   |
   v
Flow validation / compiler     (F2, arc-flow)
   |
   v
Typed Flow IR / bytecode       (F2, arc-flow)
   |
   v
Flow VM                        (F3)
   |
   v
Gameplay/world instructions    (F4)
   |
   v
Stable project world API       (M3.5)
   |
   v
ECS / runtime world
```

F2 owns semantic validation, typed IR, diagnostics tied to node/pin/connection IDs, and bytecode generation. F3 owns deterministic bytecode execution, per-instance state, lifecycle/input events, runtime bytecode validation, and instruction budgets. F4 owns gameplay/world operations and their integration through the stable M3.5 world API.
