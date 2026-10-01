# Renderer profiles

ARC resolves renderer settings from data instead of platform-name conditionals. The project descriptor's
`settings.renderer` field points at the renderer policy document; it defaults to `Config/Renderer.json`.

Resolution is deterministic and uses this precedence:

1. implemented engine quality-tier defaults;
2. the highest-priority matching target/device profile;
3. project overrides in the renderer policy document;
4. explicit runtime or editor overrides.

Device profiles match backend-neutral facts such as form factor, GPU class, logical processor count, system and GPU
memory, and executable adapter features. `platform_family` is diagnostic only and cannot be used as a profile
predicate. Equal-priority matches prefer the profile with more predicates, then the lexicographically smallest ID.

## Document example

```json
{
  "format": "arc-renderer-profile",
  "formatVersion": 1,
  "deviceProfiles": [
    {
      "id": "handheld-integrated",
      "priority": 100,
      "match": {
        "formFactor": "handheld",
        "gpuClass": "integrated",
        "maximumSystemMemoryMiB": 12288,
        "requires": ["computeShaders", "storageBuffers"]
      },
      "settings": {
        "quality": "low",
        "tiers": { "cpu": "constrained", "gpu": "constrained", "memory": "constrained" },
        "minimumRenderScale": 0.5,
        "virtualGeometry": {
          "projectedError": 2.5,
          "gpuCacheMiB": 256,
          "cpuCacheMiB": 128,
          "requestLimit": 512,
          "computeCrossoverPixels": 4,
          "hardwareCrossoverPixels": 12
        },
        "textureStreaming": {
          "gpuBudgetMiB": 384,
          "cpuBudgetMiB": 96,
          "uploadBudgetMiB": 24,
          "requestLimit": 512
        }
      }
    }
  ],
  "overrides": {
    "antiAliasing": "taa",
    "terrain": { "geometryErrorScale": 1.25 },
    "virtualTexturing": { "physicalCacheMiB": 128 },
    "postProcessing": { "quality": 0.5 }
  }
}
```

Supported feature predicates are `computeShaders`, `storageBuffers`, `storageImages`, `descriptorIndexing`,
`virtualGeometryCompute`, `virtualGeometryIndexed`, `virtualGeometryMeshShader`, `meshShaders`, `rayTracing`,
`sparseResources`, and `virtualTextures`. Unknown predicates and invalid ranges reject the document with a field-level
diagnostic.

The existing flat editor settings (`renderer.qualityTier` and `renderer.antiAliasing`) remain accepted. Template files
that contain top-level `quality` also remain compatible.

Resolved GPU/CPU caches, request limits, hybrid crossover values, terrain scaling, the selected device profile, and
fallback reasons are exposed through editor renderer diagnostics. Adapter memory can conservatively clamp configured
budgets, but a profile cannot enable a renderer path that the active backend did not advertise as executable.
