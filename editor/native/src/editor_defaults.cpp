#include <arc/editor/editor_defaults.h>

namespace arc::editor::defaults
{
static_assert(fallback_mesh_radius > 0.0f);
static_assert(imported_mesh_fit_size > 0.0f);
static_assert(native_viewport_frame_interval.count() > 0);
}
