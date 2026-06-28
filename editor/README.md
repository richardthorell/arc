# ARC Editor

This folder contains the first ARC editor shell. It is a cross-platform C++
application that links the engine in-process, uses SDL for window/input hosting,
and uses Dear ImGui docking for editor panels.

Build it from the repo root with:

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DARC_BUILD_EDITOR=ON
cmake --build build --config Release --target arc_editor --parallel
```

Or build it if needed and run it with:

```bash
python run_editor.py
```

Third-party dependencies are configured from the shared root `third_party/`
folder so editor and engine dependencies can use the same dependency policy.
