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

## F4 world runtime contract

F4 raises the Flow bytecode format to version 2 and adds gameplay/world instructions to the VM. World access is routed exclusively through the stable M3.5 `game_world_api_v1`; Flow does not receive or retain `ecs::world` pointers.

A `vm_world_context` is supplied per event dispatch and contains the borrowed M3.5 world table plus the owning entity when one exists. The world table is deliberately **not** stored on `vm_instance`, because the M3.5 bridge is valid only for the host callback that supplied it. Hosts therefore pass the current world context into Begin Play, End Play, Tick, Fixed Tick, and Input Action dispatches.

F4 bytecode supports:

- `Self Entity`
- create and destroy entity
- entity alive checks
- has/remove core component
- get/set Name
- get/set Transform using `vec3` position, `vec4` quaternion rotation, and `vec3` scale slots
- get/set Tag
- get/set Active

Flow entity values preserve both immediate M3.5 entity handles and deferred entity targets. A deferred entity returned by `create_entity` can therefore be passed directly into structural writes such as Set Name, Set Transform, Set Tag, Set Active, Remove Component, or Destroy Entity before the scheduler phase boundary. Operations that require committed world state, such as reads and alive/component checks, require an immediate entity.

World instructions consume the same per-dispatch instruction budget as control-flow instructions, so gameplay operations cannot bypass the VM's runaway-execution guard.

World failures are explicit VM outcomes. Dispatch reports `world_unavailable` when no usable world callback is supplied and `world_operation_failed` when the M3.5 operation itself cannot be completed. Source-node mapping is preserved so runtime diagnostics can later jump back to the authoring node.

F4 establishes the runtime instruction set only. The F2 compiler continues to emit the existing authored node set in this milestone; F5 adds the editor/compiler gameplay nodes that lower to these world opcodes. This keeps the VM/M3.5 boundary testable before expanding the visual language.

## F5.1 core gameplay authoring

F5.1 exposes the first world operations as authored Flow nodes and teaches the compiler to lower them to the F4 runtime instruction set. The editor node catalog includes:

- `Self Entity`
- `Is Entity Alive`
- `Get Name` / `Set Name`
- `Get Tag` / `Set Tag`
- `Get Active` / `Set Active`
- `Get Transform` / `Set Transform`
- Boolean, String, Vector3, and Vector4 literal nodes

Literal values are edited directly on the node and remain ordinary authored graph data. Gameplay inputs stay strongly typed; for example, a Vector3 cannot be connected to a quaternion/Vector4 pin.

Gameplay read nodes are explicit execution nodes in F5.1. This makes read ordering deterministic without introducing an expression scheduler yet: a graph that reads Active before branching wires `Get Active -> Branch` through both execution and value connections. `Self Entity` is a pure value source; the compiler injects a source-mapped `self_entity` prelude for event paths that consume it so authors do not need to put Self on the execution chain.

Required gameplay value inputs are compiler-validated. Missing entity/value connections produce `FLOW_REQUIRED_INPUT` diagnostics tied to the node and pin rather than deferring the error until VM execution.

F5.1 deliberately does not expose structural entity mutation. `Create Entity`, `Destroy Entity`, `Has Component`, and `Remove Component` remain the next F5.2 slice so their deferred-entity behavior can be exercised separately through authored graphs.

## F5.2 structural entity authoring

F5.2 exposes the structural F4 world instructions as authored Flow nodes: Create Entity, Destroy Entity, Has Component, and Remove Component. Has/Remove currently target ARC core components (`Name`, `Transform`, `Tag`, and `Active`) through an inline component selector.

Create Entity writes the M3.5 entity target into a typed Flow entity slot. During scheduled gameplay this may be a deferred target; downstream structural nodes and existing setters consume that same slot without forcing a premature ECS commit. This makes graphs such as `Begin Play -> Create Entity -> Set Name -> Set Transform` valid within one dispatch.

Reads that require committed world state, including Has Component and the existing Get/Alive nodes, still require an immediate entity at runtime. Flow preserves that M3.5 distinction rather than silently resolving deferred entities.

## Runtime boundary

The editable graph remains source authoring data:

```text
.arcflow
   |
   v
Flow validation / compiler     (F2 + F5 authoring lowering, arc-flow)
   |
   v
Typed Flow IR / bytecode       (F2/F5)
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

F2 owns the base compiler, typed IR, diagnostics, and bytecode generation. F3 owns deterministic bytecode execution, per-instance state, lifecycle/input events, runtime bytecode validation, and instruction budgets. F4 owns gameplay/world operations and their integration through the stable M3.5 world API. F5 exposes those operations as authored Flow nodes and compiler lowering: F5.1 covers core reads/writes and F5.2 adds structural entity creation, destruction, and core-component mutation while preserving deferred targets.
