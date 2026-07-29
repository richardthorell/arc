---
name: arc-editor-gateway
description: Use the running ARC Editor AI Gateway to inspect scenes, control and capture the viewport, diagnose rendering, wait for live events, and perform user-approved transactional scene edits.
---

# ARC Editor Gateway

Use this skill when an agent needs to interact with a running ARC editor rather
than only reading or modifying repository files. The gateway is the supported
boundary for live scene inspection, viewport debugging, renderer diagnostics,
and validated in-memory scene edits.

## Core rules

1. Start with discovery and `gateway.status`; never guess the active port or
   reuse a token from an earlier editor launch.
2. Read before acting. Resolve persistent entity GUIDs and the current
   `sceneRevision` before viewport or edit workflows.
3. Prefer `viewport.debug` (`arc_debug_viewport`) for renderer investigations.
   It configures, settles, captures, and diagnoses one coherent frame sequence.
4. Use `events.wait` instead of aggressive polling when waiting for a scene,
   frame, selection, or diagnostic change.
5. Scene writes require explicit user approval in the editor and one active
   transaction. Carry the newest `sceneRevision` through every write.
6. Verify the result before committing. Cancel the transaction when the result
   is incorrect, incomplete, or based on stale state.
7. The gateway cannot save scenes, execute scripts/processes, or read arbitrary
   files. Do not attempt to bypass those boundaries.
8. Never print, log, commit, or otherwise expose the per-launch bearer token.

## Discovery and connection

The editor listens on a random localhost port and writes a user-only discovery
file named `active.json`. The **AI Gateway** editor panel shows the exact file
path, MCP Streamable HTTP endpoint, OpenAPI endpoint, and bundled `arc-mcp`
stdio command.

The discovery document has this shape:

```json
{
  "protocolVersion": 1,
  "endpoint": "http://127.0.0.1:<port>",
  "mcpEndpoint": "http://127.0.0.1:<port>/mcp",
  "rpcEndpoint": "http://127.0.0.1:<port>/rpc/v1",
  "openApiEndpoint": "http://127.0.0.1:<port>/openapi.json",
  "token": "<per-launch-secret>",
  "pid": 12345,
  "startedAt": "<ISO-8601>"
}
```

The file is removed when the editor shuts down. Re-read it after every editor
restart.

### Authentication and identity

Send these headers on HTTP, JSON-RPC, MCP Streamable HTTP, SSE, and artifact
requests:

```text
Authorization: Bearer <discovery.token>
X-Arc-Client-Id: <stable-agent-id>
X-Arc-Client-Name: <human-readable-agent-name>
```

`X-Arc-Client-Id` may contain letters, numbers, `.`, `_`, and `-`, with a
maximum length of 80 characters. The client name is truncated to 120
characters. `X-Arc-Token` is accepted as an alternative to the bearer header,
but bearer authentication is preferred.

The server accepts at most 1 MiB per request and rate-limits each client ID to
120 requests per minute. It validates the active localhost `Host` header and
rejects untrusted browser origins.

### Transport choice

Prefer transports in this order:

1. **MCP** when the agent runtime supports it. Use the exact bundled `arc-mcp`
   command shown in the editor panel, or connect Streamable HTTP to `/mcp`.
2. **JSON-RPC 2.0** for a transport-neutral programmatic client.
3. **Direct HTTP** for simple integrations and diagnostics.
4. **SSE `/events`** for pushed gateway and editor events.

The bundled stdio bridge supports:

```text
ARC_AI_GATEWAY_DISCOVERY=<path-to-active.json>
ARC_AI_CLIENT_ID=<stable-agent-id>
ARC_AI_CLIENT_NAME=<human-readable-name>
```

Use the command displayed by the editor rather than hardcoding its packaged
path.

### JSON-RPC request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "scene.overview",
  "params": {}
}
```

Send it to `discovery.rpcEndpoint`. Successful responses contain `result`;
rejected operations return JSON-RPC error code `-32000` with a message.

### Direct HTTP request

Send the operation parameters as the JSON request body to its route. The
response shape is:

```json
{
  "result": {}
}
```

The generic `POST /api/v1/invoke` endpoint accepts:

```json
{
  "method": "viewport.state",
  "params": {}
}
```

Fetch `/openapi.json` at session start when generating a client. The document's
`x-arc-methods` array is the runtime operation catalog.

## State and revision model

Most results include:

```json
{
  "sceneRevision": 12,
  "worldEpoch": 3,
  "frameRevision": 481
}
```

- `sceneRevision` changes as authoritative scene state changes. Every mutating
  call must use the revision returned by the latest successful operation in the
  same workflow.
- `worldEpoch` identifies the loaded world. When it changes, discard cached
  entity snapshots, edit sessions, approvals, capture assumptions, and GUID
  conclusions that may belong to the previous world.
- `frameRevision` tracks viewport/render progress. Use it when waiting for a
  completed frame.
- `eventSequence` tracks gateway events and is used with `events.wait`.

A stale revision is a signal to stop writing, re-read current state, reassess
the requested change, and begin a new transaction if still appropriate. Do not
blindly retry a write with a newer revision.

## Recommended workflows

### Inspect a scene

1. Call `gateway.status` or `GET /api/v1/status`.
2. Call `scene.overview` for hierarchy, scene state, and revisions.
3. Use `scene.findEntities` to locate candidate GUIDs.
4. Use `scene.getEntity` for full component state.
5. Call `scene.componentSchemas` before interpreting or patching component
   fields.
6. Use `scene.spatialQuery` when names are insufficient.

### Debug the viewport or renderer

1. Read `viewport.state`.
2. Call `viewport.debug` with any camera move, render options, capture channels,
   and pixel samples in one request.
3. Review `requested`, `effective`, `mismatches`, `transition`, `renderer`,
   `capture.analysis`, and `anomalies`.
4. Use `viewport.inspectPixel` with the returned `captureId` for exact values.
5. Use `diagnostics.get` when broader scene, renderer, history, and recent host
   logs are needed.
6. Retain a `captureId` and use `viewport.compare` after a later change for a
   per-channel regression comparison.

For a dark, clipped, missing, or visually incorrect frame, request at least
`color`, `depth`, `objectId`, and `normals`. Add `sceneColor`, `baseColor`,
`materialProperties`, or `emissive` to isolate output, lighting, material, and
emission problems. Toggle debug visualizations one at a time when narrowing the
cause.

### Perform an approved scene edit

1. Read the target entity, component schemas, assets if needed, and current
   `sceneRevision`.
2. Call `edit.request` with a concrete label describing the intended change.
3. The user approves or denies the request in the editor. Do not spam repeated
   requests while one is pending.
4. After the user or editor indicates approval, call `edit.begin` with the
   latest `expectedSceneRevision`. A missing pending request in public status is
   not proof of approval because it may also have been denied or expired.
5. Call `edit.apply` one or more times. After each success, replace the local
   expected revision with the returned `sceneRevision`.
6. Read back the affected entities and, for visual changes, capture or debug the
   viewport.
7. Call `edit.commit` only when verification succeeds. Otherwise call
   `edit.cancel`.

Approval and inactive edit authority expire after approximately 15 minutes.
Only one edit transaction can be active, and another client may temporarily own
the viewport lease. Wait or cancel cleanly rather than fighting a lease.

## Operation catalog

All transports resolve to the operation names below. MCP combines commit/cancel
and undo/redo into paired tools, while JSON-RPC and HTTP expose the individual
operations.

| Operation | MCP tool | Direct HTTP route |
| --- | --- | --- |
| `gateway.status` | none; use HTTP or JSON-RPC | `GET /api/v1/status` |
| `scene.overview` | `arc_scene_overview` | `POST /api/v1/scene/overview` |
| `scene.findEntities` | `arc_find_entities` | `POST /api/v1/scene/find-entities` |
| `scene.getEntity` | `arc_get_entity` | `POST /api/v1/scene/entity` |
| `scene.componentSchemas` | `arc_component_schemas` | `POST /api/v1/scene/component-schemas` |
| `scene.spatialQuery` | `arc_spatial_query` | `POST /api/v1/scene/spatial-query` |
| `scene.changes` | `arc_scene_changes` | `POST /api/v1/scene/changes` |
| `assets.list` | `arc_list_assets` | `POST /api/v1/assets/list` |
| `viewport.state` | `arc_viewport_state` | `POST /api/v1/viewport/state` |
| `viewport.move` | `arc_move_viewport` | `POST /api/v1/viewport/move` |
| `viewport.setRenderOptions` | `arc_set_viewport_render_options` | `POST /api/v1/viewport/render-options` |
| `viewport.pick` | `arc_pick_viewport` | `POST /api/v1/viewport/pick` |
| `viewport.observe` | `arc_observe_viewport` | `POST /api/v1/viewport/observe` |
| `viewport.debug` | `arc_debug_viewport` | `POST /api/v1/viewport/debug` |
| `viewport.inspectPixel` | `arc_inspect_viewport_pixel` | `POST /api/v1/viewport/inspect-pixel` |
| `viewport.compare` | `arc_compare_viewport_captures` | `POST /api/v1/viewport/compare` |
| `diagnostics.get` | `arc_diagnose_viewport` | `POST /api/v1/diagnostics` |
| `events.wait` | `arc_wait_for_event` | `POST /api/v1/events/wait` |
| `edit.request` | `arc_request_edit_access` | `POST /api/v1/edit/request` |
| `edit.begin` | `arc_begin_edit` | `POST /api/v1/edit/begin` |
| `edit.apply` | `arc_apply_edit` | `POST /api/v1/edit/apply` |
| `edit.commit` | `arc_finish_edit` with `action: "commit"` | `POST /api/v1/edit/commit` |
| `edit.cancel` | `arc_finish_edit` with `action: "cancel"` | `POST /api/v1/edit/cancel` |
| `history.undo` | `arc_history` with `action: "undo"` | `POST /api/v1/history/undo` |
| `history.redo` | `arc_history` with `action: "redo"` | `POST /api/v1/history/redo` |

## Request reference

### Scene reads

#### `scene.overview`

Parameters: `{}`

Returns the current scene entity page, hierarchy/document state, and revision
metadata. The overview currently requests up to 200 entities; use
`scene.findEntities` for targeted or paged reads.

#### `scene.findEntities`

```json
{
  "search": "optional name or GUID text",
  "offset": 0,
  "limit": 100
}
```

`limit` is 1 to 200 through MCP.

#### `scene.getEntity`

```json
{
  "guid": "persistent-entity-guid"
}
```

Use GUIDs for gateway identity. Do not persist or send native entity
index/generation pairs across scene changes.

#### `scene.componentSchemas`

Parameters: `{}`

Returns reflected component types, fields, revisions, and editability metadata.
Use it as the source of truth for component field names and value shapes.

#### `scene.spatialQuery`

```json
{
  "kind": "raycast | nearby | bounds | frustum",
  "origin": [0, 0, 0],
  "direction": [0, 0, -1],
  "center": [0, 0, 0],
  "extent": [1, 1, 1],
  "radius": 10,
  "limit": 100
}
```

Supply only the vectors relevant to the selected query. Normalize ray
directions. MCP allows a maximum limit of 500.

#### `scene.changes`

```json
{
  "sinceSceneRevision": 12
}
```

When the requested revision is current, returns an empty change set. The current
implementation otherwise requests a full snapshot and sets
`fullSnapshotRequired: true`; do not assume incremental patches are available.

#### `assets.list`

Parameters: `{}`

Use returned project-relative asset paths for validated material binding.

### Viewport and diagnostics

#### `viewport.state`

Parameters: `{}`

Returns viewport dimensions, camera, render options, performance state, and
frame/revision metadata.

#### `viewport.move`

```json
{
  "action": "orbit | look | pan | dolly | frame | place",
  "x": 0,
  "y": 0,
  "amount": 0,
  "guid": "entity-guid-for-frame",
  "position": [0, 0, 0],
  "target": [0, 0, -1],
  "waitFrames": 2
}
```

- `orbit`: use `x` and `y` deltas around the current pivot.
- `look`: use `x` and `y` to rotate in place.
- `pan`: use `x` and `y` translation deltas.
- `dolly`: use `amount` for linear camera-Z movement.
- `frame`: use `guid` to select and frame an entity.
- `place`: use absolute `position` and `target` vectors.

Viewport moves are non-persistent but acquire a short viewport lease.

#### `viewport.setRenderOptions`

```json
{
  "renderMode": "shaded | wireframe",
  "visualization": "standard",
  "overlay": "none | selectedWireframe | allWireframe",
  "shadows": true,
  "environment": {
    "sky": true,
    "fog": true,
    "terrain": true,
    "water": true,
    "vegetation": true,
    "decals": true
  },
  "waitFrames": 2
}
```

Valid visualizations:

```text
standard, albedo, opacity, worldNormal, specularity, gloss, metalness, ao,
emission, lighting, uv0, cascadeDebug, shadowMask, lightComplexity,
clusterDebug
```

The result includes requested and effective options plus `mismatches`. Treat a
non-empty mismatch list as an unsuccessful configuration even if the transport
request itself succeeded.

#### `viewport.pick`

```json
{
  "x": 640,
  "y": 360
}
```

Coordinates are output viewport pixels. The operation waits for selection to
settle and returns both the request acknowledgement and selected entity state.

#### `viewport.observe`

```json
{
  "color": true,
  "depth": true,
  "objectId": true,
  "normals": true,
  "sceneColor": false,
  "baseColor": false,
  "materialProperties": false,
  "emissive": false,
  "waitFrames": 2,
  "maxWidth": 1280,
  "maxHeight": 1080
}
```

The first four channels default to enabled. Optional channels must be requested
explicitly. Results include coherent capture metadata, image analysis, PNG
artifacts, and gzip-compressed raw artifacts.

Capture artifact records include a temporary authenticated URL. PNGs are
visualizations; use `rawArtifact` for exact float, integer, or HDR data. Artifact
URLs expire after about 10 minutes and must be requested with gateway
authentication.

#### `viewport.debug`

```json
{
  "renderOptions": {
    "visualization": "worldNormal",
    "shadows": true
  },
  "camera": {
    "action": "frame",
    "guid": "entity-guid"
  },
  "capture": {
    "color": true,
    "depth": true,
    "objectId": true,
    "normals": true,
    "sceneColor": true
  },
  "samplePixels": [
    { "x": 640, "y": 360 }
  ],
  "baselineCaptureId": 123,
  "waitFrames": 2
}
```

All sections are optional. `waitFrames` is 1 to 120 and defaults to 2. The
operation returns the camera/render transition, effective settings, renderer
state, capture, optional baseline comparison, and detected anomalies.

#### `viewport.inspectPixel`

```json
{
  "captureId": 123,
  "x": 640,
  "y": 360
}
```

Omit `captureId` to create a new basic capture. Prefer an existing capture from
`viewport.debug` so all inspected channels come from the same remembered frame.

#### `viewport.compare`

```json
{
  "baselineCaptureId": 123,
  "currentCaptureId": 124,
  "waitFrames": 2
}
```

Omit `currentCaptureId` to capture the current viewport. Returns channel
compatibility, sampled mean absolute error, and changed-pixel fractions.
Remembered captures are bounded; compare promptly rather than relying on old
capture IDs.

#### `diagnostics.get`

Parameters: `{}`

Collects scene, viewport, renderer, history, recent host logs, a basic capture,
and anomaly diagnostics. This is broader and more expensive than
`viewport.state`.

#### `events.wait`

```json
{
  "kind": "scene | frame | selection | diagnostic",
  "afterSequence": 12,
  "afterFrameRevision": 481,
  "timeoutMs": 10000
}
```

Maximum timeout is 30 seconds. For `frame`, use `afterFrameRevision`; for the
other kinds, use `afterSequence` from status or a previous event.

### Approved edits

#### `edit.request`

```json
{
  "label": "Move key light and verify shadows",
  "clientName": "Rendering debug agent"
}
```

This creates a pending request for the user. It does not grant permission by
itself. Public status lists pending requests but intentionally does not expose
approved client scopes, so wait for the user/editor approval signal before
beginning an edit.

#### `edit.begin`

```json
{
  "label": "Move key light and verify shadows",
  "expectedSceneRevision": 12
}
```

Returns an `editSessionId` and current expected revision. Only one writer may be
active.

#### `edit.apply`

```json
{
  "editSessionId": "session-id",
  "expectedSceneRevision": 12,
  "action": "setTransform",
  "value": {
    "guid": "entity-guid",
    "transform": {
      "position": [0, 10, 0],
      "rotation": [0, 0, 0, 1],
      "scale": [1, 1, 1]
    }
  }
}
```

Supported actions and value fields:

| Action | Required or relevant `value` fields |
| --- | --- |
| `create` | optional `kind` (defaults to `empty`), optional `parentGuid` |
| `rename` | `guid`, `name` |
| `setActive` | `guid`, `active` |
| `setTag` | `guid`, `tag` |
| `setMobility` | `guid`, `mobility` |
| `setTransform` | `guid`, `transform` |
| `setMaterial` | `guid`, normalized project-relative `path` from `assets.list` |
| `delete` | `guid` |
| `duplicate` | `guid` |
| `reparent` | `guid`, optional `parentGuid`, optional `preserveWorld` (defaults to true) |
| `patchComponent` | `guid`, `component`, `fields` |

`patchComponent` currently binds these component names, case-insensitively:

```text
transform, camera, directionallight, pointlight, spotlight, arealight, light,
meshrenderer, terrain, worldenvironment
```

Always inspect `scene.componentSchemas` and the entity snapshot first. Patch
only intended fields; the gateway merges them with current component state.

#### `edit.commit`

```json
{
  "editSessionId": "session-id",
  "expectedSceneRevision": 13
}
```

#### `edit.cancel`

```json
{
  "editSessionId": "session-id"
}
```

Cancel on any failed validation, unexpected read-back, user change of intent,
stale assumption, or partial operation sequence.

#### `history.undo` and `history.redo`

```json
{
  "expectedSceneRevision": 14
}
```

These require active approved edit authority even though they operate on
history. Re-read the scene after completion.

## MCP resources

The MCP server also exposes read-only resources:

| URI | Contents |
| --- | --- |
| `arc://scene/summary` | Current scene overview |
| `arc://schema/components` | Reflected component schemas |
| `arc://viewport/latest` | Current viewport state |
| `arc://diagnostics/latest` | Current diagnostics bundle |
| `arc://scene/entity/{guid}` | One entity snapshot by persistent GUID |

## Failure handling

- **Discovery file missing:** start the editor and open the AI Gateway panel.
- **401:** re-read `active.json`; the token changes every launch.
- **403 host/origin:** connect directly to the exact discovery endpoint rather
  than proxying through another host.
- **429:** reduce polling and use `events.wait` or SSE.
- **Stale scene revision:** stop the transaction, re-read state, and reassess.
- **Edit permission missing/expired:** request access once and wait for the user.
- **Another edit or viewport lease is active:** wait for it to expire or for the
  owning client/user to finish. Do not repeatedly seize control.
- **Capture ID not found:** captures and artifacts are intentionally bounded;
  create a new coherent capture.
- **Render option mismatch:** use the returned effective state and report the
  mismatch rather than assuming the requested mode was applied.
- **World epoch changed:** discard all cached scene and edit assumptions.

## Completion report

When finishing a gateway task, report:

- the editor world/scene revision inspected;
- entities and persistent GUIDs involved;
- viewport modes/captures and important anomalies;
- edits requested, approved, committed, or cancelled;
- verification performed;
- anything the gateway could not do, such as saving the scene.
