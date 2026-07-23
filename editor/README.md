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
