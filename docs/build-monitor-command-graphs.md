# Build Monitor: editor command dependency graph (BM6)

BM6 extends [BM5's CMake File API graph](build-monitor-target-graph.md) to the normal editor **Build**, **Configure**, and **Rebuild** actions powered by `arc-project`.

## Scope and native source

The `arc-project` CLI uses the project's configured `paths.build/default` directory for normal commands. For an enabled editor module, its descriptor `modules[].target` is the exact native CMake root target for the dependency subtree. The CLI's default `cmake --build` builds the generator's **all** target, which is not a normal codemodel target. Therefore, the BM6 graph shows the **editor module dependency subtree only**. Other unrelated targets built by a default "all" build are not included, and no synthetic edges are created.

- Before invoking `arc-project configure` or `build`, the main process requests `codemodel-v2` in the resolved project build directory. CMake writes its reply on configure, including an implicit first-time configure performed by `arc-project build`.
- An already configured build can load the graph immediately. A first build loads the graph when native output arrives after configure, or at command completion if no output was emitted. A completed configure can refresh a changed target graph.
- CMake single-config generators may emit a codemodel configuration with an empty name; BM6 accepts that when there is only one configuration, retaining the requested build configuration label. Multi-config generators still select the exact matching configuration.
- Native artifact events reuse the BM5 exact-file-path observer. `MyEditor.vcxproj -> .../MyEditor.dll` can mark that module as having produced output; generic `ninja`/build progress messages do not.
- `ArcBuildSnapshot.targetGraph` is optional and scoped to the command's unique `runId`. BM1 persists graph revisions and emits `target-graph-changed`; BM2 retains historical graph/observation state. Previous runs without a graph still load.
- The default build succeeding does **not** satisfy the editor module target in the graph. The CLI requests **all**, so the monitor only marks individual native artifact outputs it can verify, never guessed target completion percentages or timings.

## Safety and compatibility

- Only the active project's `paths.build/default` directory and declared enabled editor module are used. No renderer-provided build path or target name is accepted.
- Paths escaping the project and existing symlinked build directories are rejected. Missing or malformed CMake metadata is nonfatal to the actual build.
- `Clean`, `Reload`, and `Open IDE` do not present synthetic target graphs. A `Rebuild` performs its existing clean/configure/build commands, each with its own unique run record.
- The existing CMake command parser, raw logs, cancellation, and module reload behavior remain unchanged.

## Follow-up

Android builds, standalone CLI invocations outside ARC Editor, and accurate per-target start/finish times require their own platform/generator event integration. BM6 does not manufacture a graph for the entire default `all` build or per-target duration bars.
