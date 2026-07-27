# ARC Editor

`editor/` contains ARC's Electron-based authoring environment. The React
workbench communicates with the authoritative C++ host in `native/`; when the
host is unavailable, the UI can use the same typed contracts through its mock
adapter.

From the repository root, prepare or run the complete editor with:

```bash
python run_editor.py
```

Use `python run_editor.py --build-only` to build the native host and type-check
the Electron application without launching it. Pass `--no-vulkan-render` to
build the host without the Vulkan viewport backend.

Electron-only workflows can be run from this directory:

```bash
npm install
npm run dev
npm run typecheck
npm test
npm run package
```

The native host owns scene state, history, persistence, asset/material
integration, viewport input, and renderer submission. Electron owns workbench
layout, panels, inspectors, asset pickers, and user interaction.

## AI scene gateway

While the editor is running, the Electron main process starts an authenticated
AI scene gateway on a random `127.0.0.1` port. Open the **AI Gateway** panel to
copy its MCP Streamable HTTP endpoint, OpenAPI document, or bundled `arc-mcp`
stdio command. The same details and the per-launch bearer token are written to
the user-only discovery file shown in that panel.

The gateway exposes equivalent MCP, JSON-RPC, and HTTP operations for live
scene inspection, viewport navigation and capture, renderer diagnostics, and
validated in-memory edits. Scene edits require approval in the editor, use one
writer lease, and form a single cancellable/undoable history transaction.
Gateway clients cannot save scenes, execute processes or scripts, or access
arbitrary files.

For renderer debugging, prefer the atomic `viewport.debug` operation (MCP:
`arc_debug_viewport`). It applies optional camera and render-debug settings,
waits for completed viewport frames, captures coherent channels, and returns
the requested and effective state with camera, shadow, environment, timing,
pixel-analysis, and anomaly diagnostics. `viewport.inspectPixel` resolves exact
color, linear depth, ObjectID/GUID, and normal values from a remembered
capture. `viewport.compare` compares a later capture with a retained baseline.

Captures can include displayed color, pre-output scene-linear HDR color,
linear depth, ObjectID, world normal, base color, material properties, and
emissive attachments. Every channel has both a PNG visualization and a
compressed raw artifact. The shared operation catalog in
`src/main/aiGatewayContract.ts` drives JSON-RPC, direct HTTP, and OpenAPI method
names so adapter documentation cannot silently drift.
