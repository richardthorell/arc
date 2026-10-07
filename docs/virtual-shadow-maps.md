# Virtual Shadow Maps

ARC models virtual shadow maps as a backend-neutral, persistent per-world page cache. Directional lights own five
equal-resolution clipmap grids whose world scale doubles at each level. Spot lights own one perspective quadtree and
point lights own six face quadtrees. All address spaces share a physical page pool made from guarded 128 by 128 texel
tiles.

Each address space receives a deterministic dense page-table range. Entries are face-major, level-major, then
row-major within the level. One entry carries independent static and dynamic physical mappings. Address-space, view,
and physical-page generations are part of the GPU ABI, so a destroyed light or recycled cache slot cannot revive a
stale mapping. Local-light ancestors divide the page coordinate by two; directional fallback instead reprojects the
receiver through the selected coarser clip view.

`shadow_map_method::auto_select` uses virtual maps only when the renderer's resolved Ultra profile reports the entire
allocation, feedback, caster-rendering, and sampling path as executable. `virtualized` requests follow
the same safety rule. An unavailable or failed virtual path resolves to conventional cascades or the local shadow atlas;
it never resolves to an unshadowed light.

The cache stores separate static and dynamic depth layers. Coarse pages are pinned, recently used pages are protected
for 30 frames, and missing fine pages sample their nearest resident ancestor.

Directional receiver demand is generated from the previous completed HZB. The GPU samples receivers on a bounded
screen grid, reprojects them through the M1 clip records, conservatively expands page demand for filtering and camera
uncertainty, and deduplicates requests in a fixed-capacity hash table. A compact request batch is copied to a per-frame
readback buffer and consumed only after that frame's fence signals. CPU translation sorts the batch, validates address
space generations and coordinates, and rejects stale work before cache allocation. Overflow is observable and never
replaces the independently generated coarse correctness pages.

Dirty physical pages carry an immutable render token containing their address-space generation, physical-page
generation, content revision, and cache work revision. Vulkan uploads a bounded page list and culls conventional GPU
Scene instances against each page-local light frustum. Culling preserves shadow distance, render-layer, mobility,
masked-material, and two-sided semantics and emits one fixed indirect range per page. The CPU may establish the
bounded page viewport and scissor, but it never submits individual casters.

Static and dynamic page passes clear and rasterize only their scheduled guarded atlas tiles through the shared
bindless depth-material path. Page work counters are copied to per-frame readback storage. A page becomes resident
only after raster and guard readiness are confirmed, the submitting frame's fence has signaled, and the exact render token still matches; overflow, unsupported
casters, failed raster setup, and stale completions leave the page dirty for retry. Partial depth is never published.
Virtual-geometry casters remain deferred to the shared traversal work in issue #482.

M4 expands each logical page projection by four texels on each side and rasterizes the complete 136 by 136 tile.
The central 128 by 128 pixels retain the logical projection. The guard-preparation graph boundary makes the completed
depth readable before publication; it does not replicate an unrelated physical neighbour. Refreshes use a replacement
tile while retaining the previously published tile. Failed or overflowed work cannot damage the old depth, and a full
pool defers replacement rather than overwriting it. Changed projection records conservatively invalidate old mappings
and render tokens until M5 introduces scrolling/overlap reuse.

Directional lights carry stable object identity, resolved representation, VSM address identity, filter mode, strength,
and biases. The existing single shadow-producing directional-light limit remains; sorting the lighting array does not
change which light owns the shadow. Deferred and terrain forward lighting use one shared lookup. Each required
static/dynamic layer independently reprojects through coarser directional clips. Missing layers use conventional
cascades. Layer comparisons are combined before averaging bounded 1/9/25-tap filters; PCSS currently uses the 25-tap
kernel. Atlas coordinates are clamped to the selected physical tile.

The canonical compiled Slang forward/transparent and Water passes use the same directional-light record layout and
per-light routing, but currently retain conventional cascade sampling. Directional VSM auto-selection remains disabled
until these material passes also use the shared M4 lookup; resource support alone must not enable a partial path.
The engine-owned compiled material pass contract is v2 and code generation is v5. Old compiled programs must be
rebuilt rather than interpreted with the new lighting-buffer stride; authoring and package container schemas are unchanged.

The resolved renderer configuration owns the physical-pool contract. It selects D16 when depth attachment and sampled
image support are both available, otherwise D32, and derives one square atlas extent from the memory budget and
adapter image limit. The CPU cache, render graph, and backend consume that exact format, extent, and capacity; they do
not independently recalculate the pool.

Vulkan realizes the shared static/dynamic depth atlases, page table, request buffers, and feedback buffers through
render-graph passes. Resources are retired after frame completion. M4 waits for prior frame fences before mutating its
shared CPU-visible tables; it does not call device-idle. Per-frame upload storage and finer cache reuse remain optimization work.
Allocation and receiver-feedback capability facts can be reported independently, but VSM light-kind support remains
disabled until caster rendering and sampling are initialized successfully. Executable support is tracked independently
for directional, point, and spot lights so an incomplete local-light path cannot disable or accidentally enable another
topology.

Directional VSM requires bindless conventional caster tables, indirect-count drawing, HZB feedback, sampled depth,
and sufficient lighting descriptor limits. Lower-limit adapters retain conventional-only shader/layout variants.
Per-light routing remains conventional until the scene tables, caster pipeline, and VSM table uploads are ready.
Point/spot VSM and virtual-geometry casters remain unavailable. Allocation failure clears the enabled VSM feature set
and records a conventional-shadow fallback diagnostic.

The Lighting panel exposes address-space count, page capacity and residency, rendered/reused pages, evictions, parent
fallbacks, failed requests, receiver samples, raw/compacted/duplicate/stale/overflow requests, and physical memory.
It also reports accepted and rejected casters, indirect draws, overflowed pages, and stale render completions. These
values describe executed work rather than requested features.
