# Virtual Geometry benchmark corpus

The ARC benchmark executable owns a deterministic generated corpus for Virtual Geometry V2. It deliberately keeps
large third-party source assets outside the repository while making local and CI measurements comparable.

## Generated corpus

Configure the normal benchmark target, then select a scale:

```text
arc-benchmarks --baseline benchmarks/baselines.json --output benchmark-results.json \
  --virtual-geometry-scale ci
```

Available scales are:

- `ci`: 8,192 deterministic source triangles. This is correctness and trend coverage, not an absolute timing gate.
- `developer`: 131,072 source triangles for routine hierarchy and encoding comparisons.
- `massive`: 10,616,832 source triangles. This is opt-in because it intentionally stresses cook time and memory.
- `off`: skip the virtual-geometry corpus.

The result records source and cooked size, hierarchy shape, page occupancy, process peak resident memory, CPU cook and
reference-traversal timing, deterministic hierarchy/selection fingerprints, page requests, and parent fallback. GPU
timings use the same traversal/raster/material fields as `render_virtual_geometry_profile`; they remain marked
unavailable in the headless CPU runner and must never be used as an absolute CI pass/fail condition.

The CI and developer scales also run deterministic VG2.4 sequences. `deterministic-occluder` compares HZB-disabled
work with previous-HZB rejection plus current-HZB refinement after history warm-up; both traversed-node and rasterized
cluster counts must fall by at least 25%. `slow-threshold` moves projected error through the ±10% hysteresis band and
requires exactly one refine and one coarsen transition.

## External corpus

`external-corpus.example.json` is the required provenance manifest for locally acquired source assets. Copy it to an
untracked `external-corpus.json`, replace the example entry, and keep downloaded assets outside the source checkout.
Every entry must pin its original URL, source revision or content hash, license identifier, local path, and import
recipe. External assets are intentionally opt-in until the asset-library provenance work in issue #354 is complete.

The generated 10M-triangle case is the authoritative reproducible stress input; external assets supplement it with
real-world topology and material boundaries rather than replacing it.

`baseline-main.json` freezes deterministic structural and selection results from `main` before the VG2.1 hierarchy
upgrade. Timing remains per-run telemetry because CPU and GPU duration is not portable across machines.
`vg2.1-comparison.json` records the deterministic before/after result for the adjacency builder: identical near/far
visible-triangle selections, half as many root-only parent fallbacks, a 0.35% stored-page increase, and effectively
unchanged artifact size.
`vg2.4-comparison.json` records the two-phase HZB before/after counters and the slow-threshold transition result.
Local CPU timings are included only as informational context; GPU timings remain per-run telemetry.
`vg2.5-comparison.json` records the developer-corpus compute, indexed-hardware, and mesh-shader work models. All three
paths consume the same final visible-cluster selection and emit the same depth/primitive identity ABI. Indexed and mesh
commands are generated on the GPU, so CPU submission remains independent of source and visible cluster counts. The
checked-in GPU counts are deterministic workload counters; live GPU timings remain informational capture data.
