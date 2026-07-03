# arc editor2

`editor2` is the Electron-based editor shell for arc.

Phase 1 provides the desktop app scaffold, React UI shell, static panel layout, secure preload bridge, and local development scripts. The native engine host and real viewport integration come in later phases.

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
- Static editor layout
- Scene hierarchy mock
- Inspector mock
- Viewport placeholder
- Console and assets panels

## Next phases

- Launch and manage `arc_host_process`
- Add typed host request/response protocol in TypeScript
- Back panels with real scene/project data
- Connect viewport resize/render option commands
- Add offscreen rendered viewport prototype
