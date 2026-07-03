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

## Workbench foundation

The workbench is now split into reusable foundation pieces:

- `app/Workbench.tsx` owns high-level editor composition and state wiring.
- `app/panelRegistry.ts` defines panel metadata, activity mappings, and default dock membership.
- `app/commandRegistry.ts` defines editor command IDs and mock/no-op command execution.
- `app/workbenchStore.ts` persists active activity and dock tabs in local storage.
- `layout/ActivityBar.tsx`, `layout/DockHost.tsx`, `layout/MenuBar.tsx`, `layout/MainToolbar.tsx`, and `layout/StatusBar.tsx` provide reusable shell components.

## Current scope

- Electron main process
- Isolated preload bridge
- React renderer
- VS Code-style workbench layout
- Activity bar and command center
- Reusable dock host with tabbed regions
- Panel registry for editor panels
- Command registry for editor actions
- Basic layout persistence
- Scene hierarchy panel backed by mock host data
- Asset browser backed by mock host data
- Inspector for selected entities and assets
- Viewport placeholder with render stats
- Bottom dock panels for content browser, console, version control, AI assistant, and profiler
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

- Expand each registered panel into its final production UI
- Launch and manage `arc_host_process`
- Replace `mockHost` with a typed host client
- Back panels with real scene/project data
- Connect viewport resize/render option commands
- Add offscreen rendered viewport prototype
