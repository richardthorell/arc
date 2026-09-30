# Platform and GPU capabilities

ARC separates immutable device facts from the features selected for a renderer instance.

- `framework::platform_capabilities` is supplied by the platform host and describes form factor, CPU and system-memory
  limits, and available platform services. Platform identity is diagnostic; feature selection must use capability fields.
- `input::input_device_capabilities` remains authoritative for each physical device.
  `input_system::capabilities()` derives a current aggregate snapshot for diagnostics and coarse feature discovery.
- `render::render_capabilities` contains adapter facts and backend-neutral limits. It includes indirect work limits,
  subgroup support, task/mesh-shader limits, atomic format support, queues, memory, and resource bounds.
- `render::render_feature_set` contains only paths deliberately enabled by the renderer. Driver support alone must not
  make an incomplete ARC path executable.

The Vulkan backend queries adapter features, properties, queues, memory heaps, and relevant format capabilities. Vulkan
types remain private to the backend. Future Direct3D and Metal backends populate the same public structures.

`query_virtual_geometry_hardware_support()` evaluates whether the raw facts can support indexed-indirect and mesh-shader
cluster rasterization. It does not enable those paths; VG2.5 owns their executable implementation.

The editor gateway diagnostics response exposes platform, connected-input, adapter, raw hardware-path, enabled-path, and
fallback information. Device/scalability profiles consume these facts rather than branching on Windows, Android, Vulkan,
or another platform/backend name.
