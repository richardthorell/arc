# ARC Unified Terrain Roadmap

This document captures the long-term terrain direction for ARC. It is intended as a durable reference for contributors, ChatGPT, Codex, and other coding agents working on terrain, virtual geometry, world building, rendering, editor tooling, or related systems.

The goal is to build **one terrain system and one terrain editor** that can achieve the same class of results as Unreal Engine's newer mesh-based terrain direction while taking advantage of ARC's clean-slate architecture. ARC should not expose separate "heightfield terrain" and "mesh terrain" workflows to users.

## Guiding Principle

> Terrain authoring format is not terrain rendering format.

A terrain may begin as a flat plane, heightmap, imported mesh, procedural source, or some combination of those. After creation, that distinction should largely disappear from the artist workflow.

The user should always work with a single `TerrainAsset` through a single Terrain Editor. The compiler decides how that terrain is evaluated, partitioned, simplified, streamed, rendered, collided, and represented on different hardware tiers.

The desired user experience is:

```text
Create -> Terrain

Choose initial source:
- Flat
- Heightmap
- Mesh
- Procedural
```

After creation, all terrain tools operate on the same asset:

```text
Sculpt | Paint | Stamp | Splines | Volume | Procedural
```

These are tools and modifiers, not different terrain types.

## Why ARC Should Not Copy Unreal's Split

Unreal's legacy Landscape system is built around heightfields and carries years of compatibility constraints. Its newer Mesh Terrain direction exists largely to escape those constraints: arbitrary topology, overhangs, caves, tunnels, variable tessellation, non-destructive modifiers, mesh inputs, and better large-world workflows.

ARC does not need to preserve a legacy Landscape-compatible authoring model. We can make the mesh-capable architecture the default terrain architecture from the start while still supporting heightmaps as a convenient source and efficient fallback.

The ARC design should therefore avoid:

- a separate `mesh_terrain_component` next to `terrain_component`
- separate editor modes for "Landscape" versus "Mesh Terrain"
- locking an asset into a heightfield representation after import
- using the editable triangle mesh as the only source of truth
- representing an entire world as a dense voxel/SDF volume
- a fixed RGBA four-layer terrain material model
- extending the dedicated heightfield quadtree indefinitely as the primary high-end runtime path

## Current ARC Foundation

ARC already has several pieces that make this roadmap practical.

### Existing terrain path

The current terrain implementation is heightfield-oriented:

- square height samples
- layer weight samples
- deterministic terrain quadtree
- projected geometric-error selection
- reusable grid patches
- neighbor LOD balancing
- stitch masks
- partial height and weight region updates
- 16-bit PNG and R16 heightmap import/export

This is useful as an input model, editing reference, validation path, and low-complexity fallback, but it should not define the long-term terrain architecture.

### Existing virtual geometry path

ARC's virtual geometry system is much closer to the desired runtime representation. It already contains concepts such as:

- topology-aware mesh clusters
- hierarchical LOD nodes
- geometric error
- independently streamable geometry pages
- compressed page payloads
- GPU-visible page tables
- page residency state
- missing-page feedback
- asynchronous loading and publication
- parent fallback
- frustum, cone, HZB, and projected-size rejection
- compute and mesh-shader raster paths
- conventional cooked LOD fallback

The terrain roadmap should build on this system rather than inventing a second Nanite-like runtime specifically for terrain.

## Target Architecture

The intended high-level model is:

```text
TerrainAsset
|
+-- Base Source
|   +-- Flat
|   +-- Heightmap
|   +-- Imported Mesh
|   +-- Procedural
|
+-- Modifier Stack
|   +-- Sculpt
|   +-- Smooth / Flatten
|   +-- Noise
|   +-- Erosion
|   +-- Remesh
|   +-- Tessellate
|   +-- Simplify
|   +-- Mesh Stamp
|   +-- Boolean
|   +-- Spline / Road
|   +-- River
|   +-- Volume Carve
|
+-- Attribute Layers / Channels
|   +-- Materials
|   +-- Biomes
|   +-- Physical Surfaces
|   +-- Foliage Masks
|   +-- Wetness / Snow / Gameplay Channels
|
+-- Build Settings
    |
    v
Terrain Compiler
    |
    v
TerrainCookedData
|
+-- Spatial Regions
+-- Virtual Geometry Hierarchy
+-- Geometry Pages
+-- Attribute / Texture Pages
+-- Collision Meshes
+-- Navigation Derived Data
+-- Fallback Conventional LODs
```

The scene-facing component should become small and asset-oriented, conceptually similar to:

```cpp
struct terrain_component
{
    asset_ref<terrain_asset> terrain;
    terrain_runtime_settings runtime;
};
```

The asset owns the authoring model. The compiler owns the runtime representation.

---

# Milestones

## Milestone 0 - Define the Unified Terrain Asset and Compiler Contract

Before adding major features to the current heightfield renderer, establish the long-term data model.

Introduce the concepts of:

### `TerrainAsset`

Owns:

- base source
- ordered non-destructive modifier stack
- named attribute channels
- build configuration
- stable region layout/versioning

A heightmap is a source node inside the terrain asset, not a separate terrain type.

### `TerrainRegion`

A stable spatial unit used for incremental evaluation, source control, streaming, and derived-data generation.

Each region should track at least:

- stable spatial coordinate or ID
- authoring bounds
- dependencies
- dirty revision
- compiled revision
- neighboring region relationships

### `TerrainCookedData`

Contains the runtime data derived from the authoring graph:

- virtual geometry references
- attribute page references
- collision data
- navigation data
- root/coarse fallback geometry
- hardware-tier fallback LODs

### Acceptance criteria

- ARC can serialize and load a `TerrainAsset` containing a heightmap source.
- Existing terrain can migrate into this model without changing visible output.
- Runtime code no longer needs to treat the serialized heightmap as the identity of the terrain object.
- The compiler boundary is explicit enough that future mesh, procedural, and volume sources can plug into the same pipeline.

---

## Milestone 1 - Compile Existing Heightfield Terrain into Virtual Geometry

This is the decisive runtime transition.

Current conceptual path:

```text
height samples
    -> terrain quadtree
    -> reusable grid patches
    -> stitch masks
```

Target path:

```text
height samples
    -> terrain region surface meshes
    -> build_virtual_mesh()
    -> virtual geometry hierarchy
    -> streamable geometry pages
```

For each terrain region:

1. evaluate the source heightfield
2. generate a high-resolution surface mesh
3. preserve deterministic region boundary topology
4. feed the mesh into ARC's virtual mesh compiler
5. generate cluster hierarchy and pages
6. render through the normal virtual geometry path
7. generate conventional LOD fallback from the same source

### Seam strategy

Region boundaries should be solved primarily at cook time rather than through permanent runtime stitch masks.

Neighboring regions should share exact compatible boundary vertices at required hierarchy levels. Temporary development fallbacks such as skirts are acceptable during implementation, but the final system should not depend on visible-overlap hacks.

### Acceptance criteria

- Existing ARC heightmaps render through virtual geometry.
- Visual shape matches the source terrain.
- No visible region or LOD cracks.
- Depth, shadows, picking, and selection work.
- Builds are deterministic.
- Conventional fallback still works on lower capability tiers.

At this milestone, new high-end terrain rendering work should stop targeting the dedicated heightfield patch renderer.

---

## Milestone 2 - Terrain-Specific Virtual Geometry Streaming

Generic virtual geometry residency already provides the base mechanism. Terrain needs additional spatial policy.

Partition cooked terrain into stable world-space regions, for example:

```text
Terrain Region
  256 m x 256 m
      |
      +-- root geometry
      +-- virtual geometry pages
      +-- attribute pages
      +-- collision
      +-- derived data
```

The exact region size should remain configurable and may evolve based on profiling.

### Required behavior

- coarse/root representation remains resident
- missing high-detail pages fall back to resident parents
- page placement favors spatial locality
- no synchronous geometry IO on the render thread
- large worlds stay inside fixed CPU and GPU residency budgets
- predictive loading considers camera movement and likely visibility

A useful priority model is conceptually:

```text
priority =
    projected error
  * screen coverage
  * visibility confidence
  * camera velocity prediction
  * gameplay importance
```

### Acceptance criteria

Use at least a synthetic 16 km x 16 km stress world and verify:

- rapid traversal does not produce holes
- memory budgets remain bounded
- high-detail data streams asynchronously
- fallback remains visually stable under missing data
- page thrashing and request overflow are measurable through diagnostics

---

## Milestone 3 - Unified Terrain Editor and Non-Destructive Sculpting

Introduce the final user-facing terrain workflow before adding exotic topology features.

The Terrain Editor should expose one asset with one ordered stack:

```text
Terrain
|- Base
|- Sculpt 01
|- Road
|- Erosion
|- Paint
```

Primary tool modes:

```text
Sculpt | Paint | Stamp | Splines | Volume | Procedural
```

### Non-destructive editing

Sculpting should not overwrite the base source permanently.

A brush stroke should modify sparse layer data associated with affected regions. The editor computes dirty bounds and schedules background rebuilds only for intersecting regions.

Conceptually:

```text
brush stroke
    -> dirty bounds
    -> affected TerrainRegions
    -> background evaluation/build
    -> publish new virtual geometry generation
```

The existing partial-region update concepts in ARC should inform this design, but the final system must work for both heightfield-like and arbitrary-topology regions.

### Editor requirements

- undo/redo
- layer visibility
- layer reorder
- layer duplication
- layer naming
- per-layer enable/disable
- incremental rebuild status
- dirty region visualization
- no whole-terrain rebuild for a local brush stroke

### Acceptance criteria

A user can:

- create one terrain
- import a heightmap
- sculpt it
- paint it
- undo/redo edits
- see only affected regions rebuild

---

## Milestone 4 - Mesh-Native Terrain and Adaptive Topology

Remove the assumption that evaluated terrain must remain a 2.5D grid.

Support base sources:

- flat terrain
- imported heightmap
- imported mesh
- procedural terrain

All produce the same `TerrainAsset`.

Add topology modifiers such as:

- remesh
- tessellate
- simplify
- mesh stamp
- project mesh

### Adaptive resolution

Resolution should become local instead of global.

Examples:

```text
rolling hill      -> sparse topology
rocky cliff       -> dense local topology
player path       -> increased local detail
flat lake bed     -> very sparse topology
```

The user should not need to increase the tessellation density of an entire world just to add detail to one cliff.

### Acceptance criteria

- terrain can start from either heightmap or mesh
- both are edited through the same tools
- local topology density responds to detail requirements
- virtual geometry cooking works from the evaluated surface
- region boundaries remain crack-free

---

## Milestone 5 - Arbitrary Topology: Caves, Tunnels, Overhangs, and Booleans

This milestone removes the remaining heightfield limitation.

Expose tools/modifiers such as:

```text
Carve
Fill
Boolean Add
Boolean Subtract
Tunnel Spline
Cave Brush
Mesh Boolean Stamp
```

### Recommended representation strategy

Do **not** make the entire world a voxel or SDF terrain.

Instead, use an implicit representation locally when a topology-changing edit requires it:

```text
surface mesh
    -> topology-changing modifier overlaps region
    -> temporary local sparse volume / SDF
    -> boolean or carve operation
    -> surface extraction
    -> local remesh
    -> terrain surface mesh
```

This keeps normal terrain mesh-based while using robust implicit operations where they provide the most value.

Prototype and compare extraction approaches such as adaptive Dual Contouring and high-quality adaptive Surface Nets before committing to one implementation.

### Acceptance criteria

An artist can:

1. sculpt a mountain
2. draw a tunnel spline through it
3. get a real entrance
4. walk through the tunnel
5. exit on the other side

without:

- a hole mask
- a separate cave terrain actor
- a second terrain type
- manually attaching a replacement cave mesh

---

## Milestone 6 - Named Terrain Attribute and Material Channels

Move beyond fixed RGBA terrain splat weights.

Terrain should support named sparse channels such as:

```text
Grass
Rock
Mud
Snow
Wetness
ForestBiome
PhysicalMaterial
FoliageDensity
GameplaySurface
```

These channels should be tiled, streamed, and updated independently from geometry where possible.

### Important invariant

> Painting terrain should normally not trigger a geometry rebuild.

Separate:

```text
Geometry -> virtual geometry pages
Attributes -> sparse/virtual terrain attribute pages
```

Materials consume world-space terrain channels.

The same channels can also drive:

- physical materials
- foliage placement
- procedural scattering
- wetness and snow
- biome systems
- gameplay surface classification

### Acceptance criteria

- arbitrary named channels
- sparse updates
- independent attribute streaming
- material painting does not rebuild geometry unless a modifier explicitly changes topology

---

## Milestone 7 - Splines, Roads, Rivers, and Procedural Modifiers

Splines should be general terrain modifier sources rather than isolated bespoke systems.

A spline may represent:

```text
Road
River
Trail
Cliff
Trench
Ridgeline
Tunnel
Retaining Wall
```

Example road stack:

```text
Road Spline
|- flatten modifier
|- local remesh modifier
|- road material channel
|- shoulder material channel
|- foliage exclusion channel
|- optional road mesh generation
```

Example river stack:

```text
River Spline
|- carve
|- smooth banks
|- wetness paint
|- sediment paint
|- water-system binding
```

Procedural terrain modifiers should include at least:

- noise
- terracing
- slope filters
- curvature filters
- thermal erosion
- hydraulic erosion
- biome generators

These should participate in the same ordered modifier stack rather than requiring a separate terrain-only graph system.

### Acceptance criteria

- spline edits invalidate only affected regions
- road and river workflows are non-destructive
- procedural layers can be reordered with sculpt/paint layers
- generated channels can feed materials and foliage

---

## Milestone 8 - Collision, Navigation, Foliage, Water, and World-System Integration

A changed terrain region should fan out into independently generated derived data:

```text
Terrain region changed
    |
    +-- render geometry
    +-- collision proxy
    +-- navmesh dirty tiles
    +-- foliage placement
    +-- water interaction
    +-- GI / ray-query representation
    +-- distance/query representation
```

All expensive work should be asynchronous and region-local.

### Independent error budgets

Visual, collision, and navigation representations should not be forced to use identical detail.

For example:

```text
Visual      -> millimeter to centimeter effective detail where needed
Collision   -> gameplay-dependent decimeter detail
Navigation  -> independent nav-tile representation
```

Editing a small region must not rebuild collision or navigation for the entire terrain asset.

### Acceptance criteria

- region-local collision rebuild
- region-local nav invalidation
- foliage responds to changed attribute channels
- water and terrain modifier integration is stable
- runtime queries work across caves and arbitrary topology

---

## Milestone 9 - Production Terrain Workflow and Diagnostics

Finish the system around the artist and large-team workflow.

Add tooling such as:

- layer profiler
- build-to-this-layer
- solo layer
- freeze layer
- bake/collapse layers
- per-region build time
- dirty-region visualization
- LOD visualization
- cluster visualization
- streaming visualization
- material-channel visualization
- collision preview
- navigation preview
- residency and page-request diagnostics

### Source-control-friendly authoring

One brush stroke should not rewrite one giant terrain binary.

Prefer a regionized source layout conceptually similar to:

```text
Mountain.terrain
Mountain/
    regions/
        12_17...
        12_18...
    layers/
        sculpt_01/...
        road_04/...
```

Expensive evaluated and cooked results belong in derived-data/cache systems rather than source control where possible.

### Acceptance criteria

- large terrain edits produce localized source changes
- modifier cost is visible to artists
- rebuild bottlenecks are diagnosable
- terrain data is practical in multi-user source-control workflows

---

# Runtime Representation Policy

There should still be only one conceptual terrain even when hardware capabilities differ.

The compiler can produce multiple runtime representations from the same `TerrainAsset`:

```text
                    TerrainAsset
                        |
                     compiler
                        |
            +-----------+-----------+
            |                       |
     Virtual Geometry        Conventional LODs
     high-end path            lower-end fallback
            |                       |
  Compute / Mesh Shader          indexed draws
```

The artist should not select or maintain these representations manually.

Mesh shaders must remain an optimization, not an architectural requirement. ARC's compute path should remain a first-class implementation for virtual terrain geometry.

# Architectural Rules for Future Work

When implementing terrain-related changes, preserve these rules unless this document is intentionally revised:

1. **One terrain asset type.** Do not create parallel heightfield and mesh terrain product concepts.
2. **One Terrain Editor.** Different capabilities appear as tools/modifiers, not separate editors.
3. **Non-destructive authoring.** Base sources remain recoverable; modifiers are ordered and editable.
4. **Authoring is separate from runtime.** Runtime data is compiled and may vary by hardware tier.
5. **Virtual geometry is the primary high-end runtime destination.** Avoid building a second independent terrain virtualization stack.
6. **Local edits cause local rebuilds.** Region granularity is fundamental to editor responsiveness and large-world scalability.
7. **Topology can become arbitrary.** Heightfields are convenient inputs, not a permanent limitation.
8. **Use implicit/SDF representations locally, not globally, unless future profiling proves otherwise.**
9. **Geometry and attributes stream independently.** Painting should not normally rebuild geometry.
10. **Derived systems have independent detail budgets.** Rendering, collision, navigation, foliage, and queries should not share unnecessary resolution.
11. **Backend neutral first.** Vulkan is the first implementation, not the terrain architecture.
12. **Graceful hardware fallback is automatic.** The user authors one terrain and ARC chooses the runtime path.

# Immediate Implementation Priority

From the current ARC state, the next terrain work should be prioritized as:

1. **`TerrainAsset` and compiler contract**
   - establish source/layer/region/cooked-data separation

2. **Heightmap -> region mesh -> `build_virtual_mesh()`**
   - move current terrain rendering through the virtual geometry stack

3. **Unified Terrain Editor with layered sculpting and incremental region compilation**
   - establish the final artist workflow early

4. **Adaptive remesh + mesh source + boolean/volume modifiers**
   - reach the core arbitrary-topology capability

After these four, roads, erosion, biome painting, foliage, world integration, and production tooling should extend the same architecture rather than require another terrain rewrite.

# Reference Direction

Useful external references for concepts, not APIs to copy directly:

- Unreal Engine Mesh Terrain documentation: https://dev.epicgames.com/documentation/en-us/unreal-engine/mesh-terrain-in-unreal-engine
- Unreal Engine PCG and Mesh Terrain documentation: https://dev.epicgames.com/documentation/en-us/unreal-engine/pcg-and-mesh-terrain-in-unreal-engine
- Unreal Engine Mesh Terrain access/editor documentation: https://dev.epicgames.com/documentation/en-us/unreal-engine/accessing-mesh-terrain-in-unreal-engine

ARC should adopt the useful ideas—arbitrary topology, non-destructive modifier stacks, local variable detail, regionized processing, and compiled runtime sections—while keeping the product model simpler: **one terrain system from creation through shipping.**
