# Multi-player and client/server Play architecture

ARC's future multi-player Play-in-Editor (PIE) model extends the existing isolated Play World and project-module boundaries without making the common single-player authoring loop pay for networking concerns.

## Design goals

- Keep one-player Play as the default path: one runtime world, one local player, no transport, and no server process unless explicitly requested.
- Treat every PIE world/process as an isolated runtime instance. Authoring World state is copied into a session snapshot and is never used as gameplay state.
- Route physical devices through logical player contexts before gameplay consumes input. Gameplay and Flow must not depend on keyboard/gamepad device identity.
- Make client/server role explicit at the session boundary and project-module boundary rather than adding networking branches throughout authoring-world code.
- Leave deterministic hooks for latency, jitter, packet loss, and future transport simulation without requiring those hooks in local single-player Play.

## Session topology

A PIE session owns a stable session ID and one or more runtime instances. Each instance has its own instance ID, role, world, player set, module generation, log stream, and lifecycle.

```text
Authoring World
      |
      | immutable start snapshot
      v
PIE Session
  |- Runtime Instance 0: Standalone/Client, Play World 0, local player 0
  |- Runtime Instance 1: Client,            Play World 1, local player 1
  `- Runtime Instance S: Server,            Play World S, no local player
```

The first implementation remains the existing single runtime instance. Multi-player support generalizes the owner from "the Play World" to a collection without changing world semantics. Every runtime instance receives paired Begin/End Play and is stopped independently; the session is complete only after all instances have ended.

### Runtime roles

Use explicit roles at launch/runtime boundaries:

- `Standalone`: the current simple local Play path. It has one runtime world and may have one or more local logical players, but no network peer is required.
- `Client`: a runtime world that owns zero or more local logical players and connects to a server endpoint.
- `Server`: an authoritative runtime world with no editor physical-input ownership. It loads the Server project module role when available.

Role is immutable for an instance lifetime. Changing topology recreates affected instances rather than mutating a running world's role.

## Logical players and input

Physical input remains sampled by the editor/native host, but device ownership is resolved before a runtime instance receives semantic actions.

A session-level player assignment maps:

```text
physical device(s) -> logical player ID -> runtime instance ID -> input context stack
```

A logical player can own multiple devices (for example keyboard + mouse, or gamepad + gyro), while each physical device has at most one active local-player owner unless an explicit shared-device policy is configured. Device connect/disconnect changes assignment state; it does not change entity identity or world ownership.

The existing player `0` path is the compatibility/default mapping. Projects that do not opt into multi-player continue to receive exactly that logical player and need no networking configuration. Flow and gameplay consume semantic actions for a logical player, never raw host-device routing.

Split-screen is a presentation concern layered on the same runtime instance: multiple local logical players can select separate player/camera entities and viewport rectangles while sharing one Play World. Multiple client worlds instead use separate runtime instances, even when rendered in one editor window.

## World and process ownership

The session topology must not assume all instances live in one process. The orchestration contract treats in-process and child-process instances uniformly.

Initial development can host multiple isolated worlds in the editor native host when safe. Client/server validation should also support launching separate runtime processes through the same packaged/runtime entry boundary used by Play Standalone. Separate processes are required whenever module/global-state isolation or realistic client/server behavior cannot be guaranteed in-process.

Each instance owns:

- exactly one isolated runtime world;
- one selected project-module role/generation;
- an independent lifecycle and stop reason;
- an independent log/diagnostic channel tagged with session and instance IDs;
- zero or more local logical players;
- optional network endpoint metadata.

No runtime instance may hold mutable references into the Authoring World. Editor inspection targets an instance ID plus stable runtime entity/component identity.

## Client/server module roles

The existing project module descriptor already distinguishes Editor, Runtime, and Server roles. PIE uses that boundary directly:

- Standalone and Client instances load the Runtime role.
- Server instances load the Server role.
- The Editor role remains editor-only and is never treated as a gameplay server.

A topology request is rejected before world startup if a required role is unavailable or incompatible. Module reload/restart policy remains generation-based per instance; a schema change that requires restart recreates the affected runtime instances from a clean session snapshot rather than migrating opaque runtime state.

## Network emulation hooks

Network emulation belongs at the PIE transport boundary, not in ECS, authoring data, Flow, or gameplay systems. A per-link policy can later describe disabled/default behavior or controlled latency, jitter, loss, duplication, reordering, and bandwidth limits.

```text
Client World -> project networking -> PIE transport link -> emulation policy -> Server World
```

The default policy is pass-through and imposes no simulated network behavior. Emulation configuration is transient editor/session state and must not be serialized into scenes or prefabs. Runtime/gameplay code observes normal transport behavior and does not branch on "running in editor" to implement emulation.

## Lifecycle and failure model

Starting a topology is transactional at the session level:

1. Validate requested roles, player assignments, build/module compatibility, and launch configuration.
2. Capture one deterministic authoring snapshot for the session.
3. Create server instances first when clients require them, then create clients/standalone instances from the same authored baseline.
4. Publish an instance as running only after its module and Play World start successfully.
5. If required startup fails, stop every instance already started and report the failure tagged with the intended instance/role.

Stopping the session sends End Play to every running instance and releases each world/module generation independently. A crashed child process is reported as an instance failure; it must not cause the Authoring World to adopt partial runtime state.

## Authoring isolation

Networking assumptions must not leak into core authoring data:

- Scene/prefab serialization describes authored entities/components, not PIE instance IDs, ports, or process roles.
- Runtime-only connection state and replicated snapshots live only in runtime worlds.
- Keep Simulation Changes, when implemented, compares a selected runtime world against the Authoring World using its explicit supported-change policy; it never copies transport/session state.
- Play From Here is an instance-local transient spawn override and does not mutate the authored scene.

## Incremental implementation path

1. Generalize the current single Play session owner to expose stable session/instance identity while preserving the one-instance behavior.
2. Generalize semantic input dispatch from hard-coded logical player `0` to explicit player contexts and device assignment.
3. Support multiple local players in one Standalone world; add per-player camera/viewport selection without networking.
4. Add multiple isolated client runtime instances under one PIE session.
5. Add Server-role instances and client/server orchestration through the runtime launch boundary.
6. Add optional per-link network emulation and editor controls.

Each step must keep the single-player configuration as the zero-extra-configuration path and must preserve the existing isolated-world lifecycle.

## Validation checklist

- Single-player Play still starts one isolated world with logical player `0` and no networking setup.
- Two local players can have independent semantic input contexts without exposing physical device IDs to gameplay.
- Multiple client instances never share mutable ECS/world state.
- Server instances use the Server module role and do not consume editor physical input.
- Begin/End Play are paired for every runtime instance on normal stop and partial-start failure.
- Logs, inspection, stop, and restart operations identify both PIE session and runtime instance.
- Network emulation can be disabled completely and remains outside authored scene/prefab state.
- No client/server or PIE process metadata is required by the Authoring World.