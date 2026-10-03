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
allocation, feedback, caster-rendering, sampling, and contact-shadow path as executable. `virtualized` requests follow
the same safety rule. An unavailable or failed virtual path resolves to conventional cascades or the local shadow atlas;
it never resolves to an unshadowed light.

The cache stores separate static and dynamic depth layers. Coarse pages are pinned, recently used pages are protected
for 30 frames, and missing fine pages sample their nearest resident ancestor.

The resolved renderer configuration owns the physical-pool contract. It selects D16 when depth attachment and sampled
image support are both available, otherwise D32, and derives one square atlas extent from the memory budget and
adapter image limit. The CPU cache, render graph, and backend consume that exact format, extent, and capacity; they do
not independently recalculate the pool.

Vulkan realizes the shared static/dynamic depth atlases, page table, request buffer, and feedback buffer through
render-graph passes. Resources are retired after frame completion and ordinary rendering never waits for device idle.
The Vulkan backend must keep the VSM capability facts disabled until its complete caster-render and sampling pipelines
are initialized successfully. Executable support is tracked independently for directional, point, and spot lights so
an incomplete local-light path cannot disable or accidentally enable another topology.

The Lighting panel exposes address-space count, page capacity and residency, rendered/reused pages, evictions, parent
fallbacks, failed requests, and physical memory. These values describe executed work rather than requested features.
