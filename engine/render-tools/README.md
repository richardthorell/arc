# ARC Render Tools

`arc-render-tools` contains source-authoring services that are intentionally
absent from Shipping runtimes. It compiles source-authored Slang modules and
deterministically lowered material graphs into backend-neutral ARC shader
packages. Runtime render modules consume package reflection and target
bytecode; they never invoke a source compiler.

The tool requires the exact Slang version declared by `pinned_slang_version`.
The compiler executable can be selected with `ARC_SLANGC_EXECUTABLE`; otherwise
the tool searches the process path. Compile requests hash source, the recursive
include closure, entry point, target, definitions, optimization policy, compiler
fingerprint, and ARC shader-library version.

Shared source lives under `shaders/`. Templates are starting points for authored
materials and utility shaders, not privileged runtime paths. Graphs and source
modules both lower to the `ArcSurfaceData` contract and use the same compiler,
reflection, package, cache, and diagnostics path.
