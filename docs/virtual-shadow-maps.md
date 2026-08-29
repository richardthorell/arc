# Virtual Shadow Maps

ARC models virtual shadow maps as a backend-neutral, persistent per-world page cache. Directional lights own five
clip levels, spot lights own one perspective quadtree, and point lights own six face quadtrees. All address spaces
share a physical page pool made from guarded 128 by 128 texel tiles.

`shadow_map_method::auto_select` uses virtual maps only when the renderer's resolved Ultra profile reports the entire
allocation, feedback, caster-rendering, sampling, and contact-shadow path as executable. `virtualized` requests follow
the same safety rule. An unavailable or failed virtual path resolves to conventional cascades or the local shadow atlas;
it never resolves to an unshadowed light.

The cache stores separate static and dynamic depth layers. Coarse pages are pinned, recently used pages are protected
for 30 frames, and missing fine pages sample their nearest resident ancestor. Address spaces and physical pages are
generation checked so light destruction, hot reload, and pool reuse cannot expose stale mappings.

Vulkan realizes the shared static/dynamic depth atlases, page table, request buffer, and feedback buffer through
render-graph passes. Resources are retired after frame completion and ordinary rendering never waits for device idle.
The Vulkan backend must keep the VSM capability facts disabled until its complete caster-render and sampling pipelines
are initialized successfully.

The Lighting panel exposes address-space count, page capacity and residency, rendered/reused pages, evictions, parent
fallbacks, failed requests, and physical memory. These values describe executed work rather than requested features.
