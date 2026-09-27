#include <arc/editor/host_protocol.h>

// Keep the protocol implementation in one translation unit so additions to the
// command/query variants produce one authoritative set of JSON entry points.
// Extend the existing mesh-renderer JSON object at its unique material-backed
// field without carrying a second copy of the protocol implementation.
#define asset_backed_material                                                                                          \
    asset_backed_material) + \
        ",\"hasMesh\":" + bool_json(mesh_renderer.has_mesh) + \
        ",\"assetBackedMesh\":" + bool_json(mesh_renderer.asset_backed_mesh) + \
        ",\"meshName\":" + quote(mesh_renderer.mesh_name) + \
        ",\"meshPath\":" + quote(mesh_renderer.mesh_path
#include "host_protocol_base.inc"
#undef asset_backed_material
