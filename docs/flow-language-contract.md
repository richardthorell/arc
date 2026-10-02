# Flow language contract

Flow is ARC's typed, deterministic gameplay graph language. This document records the language/runtime contract that editor authoring, compilation, bytecode execution, and reusable composition must preserve.

## Typed values

Flow values are explicitly typed. The shared runtime supports booleans, integers, floats, vectors, strings/names, entities, and component-facing values. Graph variables, graph interfaces, and function interfaces carry their type through compilation; invalid or incompatible operations must fail compilation or execution rather than rely on implicit dynamic coercion.

Numeric conversion is explicit. Integer-to-float and float-to-integer conversions are bytecode operations, so a graph's conversion behavior remains visible and deterministic.

## Execution order

Execution order is part of language semantics, not canvas layout. Outgoing execution paths are resolved from the declared node/pin semantics with deterministic fallback ordering. Moving nodes or changing connection insertion order must not change runtime behavior.

Control-flow instructions include branching, ordered sequence/fan-out, integer switching, do-once/gate state, bounded loop forms, and latent/timer operations. Their continuations are encoded explicitly in compiled artifacts.

Function and subgraph composition is synchronous unless a construct is explicitly latent. Function calls bind typed input/output slots and carry an explicit continuation; returns copy declared outputs before continuing the caller. This keeps reusable composition deterministic and prevents editor-only graph structure from leaking into runtime dispatch.

## Events and latent work

Entry points include Play lifecycle, tick/fixed-tick, semantic input actions, and custom events. Custom events use stable authored definitions while runtime dispatch uses the compiled event mapping.

Delay, retriggerable delay, and timer operations are the language's explicit latent boundary. Latent work must resume through a compiled continuation rather than retaining editor graph objects or depending on wall-clock/editor state.

## Instruction budget

Every VM execution remains subject to ARC's instruction-budget protection (`default_instruction_budget` is 4096). New control-flow, function, event, or latent constructs must participate in the same accounting; reusable composition must not provide a path around the budget.

When the budget is exhausted, execution fails safely instead of continuing an unbounded graph. Language extensions must preserve this behavior in focused runtime tests.

## Determinism requirements

A Flow language extension is ready only when:

- authored types compile to typed IR/bytecode without provider/editor-specific runtime state;
- execution order is defined by language semantics rather than visual placement or container iteration order;
- invalid type/control-flow combinations produce actionable diagnostics;
- instruction-budget protection still applies across the new construct;
- save/reload and recompilation preserve stable authored identities where the construct exposes reusable interfaces.

## Current composition boundary

Local typed functions and graph interfaces are the reusable composition boundary. They use stable IDs for authored interface values and compiled slots for runtime transport. Cross-asset composition and collections should only be added when their ownership, versioning, migration, and deterministic execution semantics are defined; they must not be introduced as untyped generic containers.

This contract is intentionally backend-neutral. The Flow editor may add richer authoring affordances, but the engine's typed compiled representation remains authoritative for runtime behavior.
