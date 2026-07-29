# ARC public API migration

This cleanup intentionally removes the pre-production compatibility surface.
Persisted GUIDs, component IDs, document fields, and package formats are
unchanged.

| Previous API | Current API |
|---|---|
| `arc::job_system` | `arc::jobs::job_system` |
| `arc::memory_system` | `arc::memory::memory_system` |
| `arc::logger` | `arc::diagnostics::logger` |
| `arc::runtime` | `arc::framework::runtime` |
| `arc::simd<T, N>` | `arc::simd::simd<T, N>` |
| `arc::scene::registry` | `arc::ecs::world` |
| `arc::scene::entity` | `arc::ecs::entity` |
| `arc::scene::entity_guid` | `arc::ecs::entity_guid` |
| `<arc/assets.h>` | `<arc/assets/assets.h>` |
| `<arc/io.h>` | `<arc/io/io.h>` |
| `<arc/persistence.h>` | `<arc/persistence/persistence.h>` |
| `material_desc` | `material_descriptor` |
| `texture_desc` | `texture_descriptor` |
| `environment_desc` | `environment_descriptor` |

Recoverable operations now use `arc::core::result<T, E>` or
`arc::core::status<E>`. Test a result with `has_value()` or an explicit Boolean
conversion, then access `value()` or `error()`.

Renderer frame submission now returns `core::status<render_submit_error>`.
Backend creation returns
`core::result<std::unique_ptr<render_backend>, render_backend_create_error>`,
and shader compilation returns
`core::result<shader_compile_output, shader_compile_error>`. The former
`submitted`, `succeeded`, and free-form factory-message fields are removed.

The Electron host presents its native surface through
`render_backend::present_surface_frame()`. Consumers no longer downcast the
backend or access Vulkan editor hooks.
