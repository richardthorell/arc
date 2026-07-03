# arc editor2

`editor2` is the Electron-based editor shell for arc.

The current shell is a VS Code-inspired 3D engine workbench with mock/no-op host data. It is structured so the renderer can later swap from the mock service to the real `arc_host_process` protocol without rewriting the UI.

## Development

Install dependencies:

```bash
npm install
```

Run the editor:

```bash
npm run dev
```

Type-check the Electron app:

```bash
npm run typecheck
```

Package locally:

```bash
npm run package
```

## Current scope

- Electron main process
- Isolated preload bridge
- React renderer
- VS Code-style workbench layout
- Activity bar and command center
- Scene hierarchy panel backed by mock host data
- Asset browser backed by mock host data
- Inspector for selected entities and assets
- Viewport placeholder with render stats
- Bottom panel with Problems, Output, Debug Console, Terminal, and Profiler tabs
- Mock/no-op host service for commands and data loading

## Mock host

Phase 3 keeps the native engine host disconnected on purpose. The renderer uses `src/renderer/src/services/mockHost.ts` to simulate:

- Project snapshot
- Scene hierarchy
- Asset list
- Console events
- Render statistics
- Entity selection
- No-op command execution

When `arc_host_process` is ready, the same UI can be pointed at a real host client with the same high-level shape.

## Next phases

- Launch and manage `arc_host_process`
- Replace `mockHost` with a typed host client
- Back panels with real scene/project data
- Connect viewport resize/render option commands
- Add offscreen rendered viewport prototype
