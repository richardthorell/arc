#pragma once

// Keep the generated/native protocol surface stable while extending the mesh-renderer
// snapshot with the authored mesh asset that feeds the runtime geometry handle.
// The base header owns the protocol declarations; this narrow token injection avoids
// duplicating that large declaration file while the protocol is split into modules.
#define asset_backed_material                                                                                          \
    asset_backed_material{};                                                                                           \
    bool has_mesh{};                                                                                                   \
    bool asset_backed_mesh{};                                                                                          \
    std::string mesh_name;                                                                                             \
    std::string mesh_path;                                                                                             \
    bool arc_mesh_snapshot_extension
#include <arc/editor/host_protocol_base.h>
#undef asset_backed_material
