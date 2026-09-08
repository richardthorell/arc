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

**Status: complete.** The implementation is intentionally renderer-independent and establishes the contracts used by later milestones.

Implemented M0 foundations include:

- one versioned `TerrainAsset` with Flat, Heightfield, Mesh, and Procedural source identities
- ordered/versioned modifier descriptors and named attribute definitions
- persistent `.terrain` assets and typed importer support
- stable authoring-region IDs, high-precision origins, dirty/compiled revisions, dependency halos, and build snapshots
- scene compatibility bridging from existing inline heightfield terrain
- owning evaluated terrain data plus the renderer-independent `TerrainSurfaceIR` boundary
- region-aware Flat and resolved-Heightfield evaluation, with registered modifier dispatch
- deterministic evaluated-surface fingerprints
- opaque content-addressed derived-artifact keys and per-region cooked manifests
- independent artifact identities for render geometry, fallback geometry, attributes, collision, navigation, destruction, and ray queries
- deterministic authoring seam ownership independent from virtual-geometry page/cluster boundaries
- a versioned runtime operation journal contract for future deformation, fracture persistence, and replication

Heightfield sample resolution deliberately remains outside the evaluator so asset streaming/import systems can supply source data without coupling the terrain authoring model to one texture/backend representation. Mesh and Procedural source identities already use the same contract; their actual surface providers are later milestones.

M0 explicitly does **not** define virtual-geometry page size, cluster layout, GPU representation, collision topology, navigation tiling, or fracture hierarchy. Those remain independent derived systems.

### Acceptance criteria

- ARC can serialize and load a `TerrainAsset` containing a heightmap source.
- Existing terrain can migrate into this model without changing visible output.
- Runtime code no longer needs to treat the serialized heightmap as the identity of the terrain object.
- The compiler boundary is explicit enough that future mesh, procedural, and volume sources can plug into the same pipeline.
- Future virtual geometry, cave/topology, and destruction systems can consume or produce `TerrainSurfaceIR`/derived artifacts without redesigning `TerrainAsset`.

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

Splines should be generic terrain modifier inputs rather than special terrain actors.

A spline may represent:

```text
Road
River
Trail
Cliff
Trench
Ridgeline
Tunnel
Retaining wall
```

Example road stack:

```text
Road Spline
|- flatten modifier
|- local remesh modifier
|- road material channel
|- shoulder material channel
|- foliage exclusion channel
`- optional road mesh
```

Example river stack:

```text
River Spline
|- carve
|- smooth banks
|- wetness paint
|- sediment paint
`- Water System binding
```

Procedural modifiers should eventually include:

- noise
- thermal erosion
- hydraulic erosion
- terracing
- slope filters
- curvature filters
- biome generation

These should be normal terrain modifiers rather than requiring a parallel terrain-specific PCG product.

### Acceptance criteria

- spline edits rebuild only intersecting terrain regions
- roads can affect geometry, material channels, and foliage masks in one stack
- rivers can carve terrain and bind to water data
- procedural modifiers can be reordered with authored modifiers

---

## Milestone 8 - Collision, Navigation, Foliage, Water, and World-System Integration

A terrain region rebuild should fan out into independent derived products:

```text
Terrain region changed
        |
        +-- render geometry
        +-- collision proxy
        +-- navigation dirty tiles
        +-- foliage placement
        +-- water interaction
        +-- GI / ray-query data
        `-- distance/query representation
```

All of these should be asynchronous where practical.

### Collision

Collision needs its own error budget. It should not automatically use the full visual triangle count.

For example:

```text
Visual detail:     millimeters to centimeters where needed
Collision detail:  centimeters to decimeters based on gameplay
Navigation:        independent tile representation
```

A local terrain edit should invalidate local collision and navigation data rather than rebuilding kilometers of world data.

### Acceptance criteria

- visual and physics complexity are decoupled
- navigation updates are spatially incremental
- foliage responds to terrain attribute changes
- rivers/water can respond to terrain edits
- ray/GI data can be independently generated or invalidated

---

## Milestone 9 - Production Terrain Workflow and Diagnostics

Finish the system around artists and large projects.

Useful Terrain Editor features:

```text
Layer profiler
Build To This Layer
Solo layer
Freeze layer
Bake / collapse layers
Per-region build time
Dirty-region visualization
LOD visualization
Cluster visualization
Streaming visualization
Material-channel visualization
Collision preview
Navigation preview
```

### Source-control layout

Terrain source data should remain regionized so a small brush stroke does not rewrite one giant binary file.

Conceptually:

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

Expensive evaluated and cooked products belong in derived-data storage/cache rather than source control.

### Acceptance criteria

- local authoring edits create local source-control changes
- expensive derived data is cacheable and reproducible
- artists can inspect what caused an expensive rebuild
- build profiling identifies expensive modifiers and regions
- CI/cook can reproduce terrain from authored inputs deterministically

---

# Runtime Scaling and Hardware Compatibility

ARC should expose one authored terrain regardless of hardware tier.

Conceptually:

```text
                       TerrainAsset
                            |
                         compiler
                            |
             +--------------+--------------+
             |                             |
     Virtual Geometry                Conventional LODs
    high-end / desktop              low-end / fallback
             |                             |
      Compute / Mesh Shader            indexed draws
```

The artist should not choose separate terrain assets for these paths.

Mesh shaders should remain an optimization rather than a hard architectural requirement. Compute-driven virtualized geometry should remain a first-class path when practical.

# Architecture Rules

These rules should be preserved as the terrain system evolves.

## Do

- keep one `TerrainAsset`
- keep one Terrain Editor
- treat heightmaps as source data
- use non-destructive ordered modifiers
- rebuild only spatially affected regions
- keep authoring and cooked representations separate
- use virtual geometry as the primary high-end runtime destination
- generate lower-tier fallbacks automatically
- stream geometry and attributes independently
- keep collision and navigation representations independent from visual detail
- use local implicit/SDF representations for topology-changing operations where useful

## Do Not

- add `mesh_terrain_component` alongside `terrain_component`
- make an entire world permanently voxel-based just to support caves
- expose render-cluster/page sizes as artist-facing terrain concepts
- extend runtime stitch masks as the long-term seam solution
- make material painting rebuild geometry unnecessarily
- require a second terrain actor for caves or tunnels
- force mesh shaders as the only viable virtual-geometry implementation
- couple collision complexity directly to visual triangle count

# Recommended Implementation Order

The highest-value near-term sequence is:

1. `TerrainAsset` and compiler contract
2. compile existing heightfield terrain into virtual geometry
3. unified Terrain Editor with non-destructive sculpting and incremental region builds
4. mesh-native adaptive topology
5. local volume/boolean operations for caves and tunnels
6. independent named attribute streaming
7. splines, erosion, roads, rivers, and procedural workflows
8. collision/navigation/foliage/world-system integration
9. production diagnostics and large-project tooling

The key architectural transition is step 2: once existing terrain renders through the normal virtual-geometry path, ARC stops investing new high-end rendering work into the dedicated heightfield patch renderer.
