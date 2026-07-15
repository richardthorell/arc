# arc editor2

`editor2` is ARC's Electron-based editor shell. It uses the native `arc_host_process` when available and retains a typed mock snapshot for disconnected development.

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

Run renderer component tests:

```bash
npm test
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
- Schema-driven Transform and Camera inspector backed by typed host snapshots
- Viewport placeholder with render stats
- Bottom dock panels for content browser, console, version control, AI assistant, and profiler
- Mock/no-op host service for commands and data loading

## Mock host

When the native engine host is unavailable, the renderer uses `src/renderer/src/services/mockHost.ts` to simulate:

- Project snapshot
- Scene hierarchy
- Asset list
- Console events
- Render statistics
- Entity selection
- No-op command execution

## Next phases

- Expand each registered panel into its final production UI
- Back panels with real scene/project data
- Connect viewport resize/render option commands
- Add offscreen rendered viewport prototype
