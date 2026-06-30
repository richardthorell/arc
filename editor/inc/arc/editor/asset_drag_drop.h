#pragma once

#include <filesystem>

namespace arc::editor
{

enum class editor_asset_kind
{
    unknown,
    directory,
    material,
    texture,
    mesh,
    shader,
    environment,
    scene
};

struct editor_asset_payload
{
    editor_asset_kind kind{ editor_asset_kind::unknown };
    std::filesystem::path relative_path;
};

inline constexpr const char* asset_path_payload_id = "ARC_ASSET_PATH";
inline constexpr const char* material_asset_payload_id = "ARC_MATERIAL_ASSET";
inline constexpr const char* texture_asset_payload_id = "ARC_TEXTURE_ASSET";

editor_asset_kind classify_asset_path(const std::filesystem::path& path, bool directory = false);

bool is_texture_asset_kind(editor_asset_kind kind) noexcept;

std::filesystem::path asset_relative_path(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path);

bool make_asset_payload(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    bool directory,
    editor_asset_payload& payload);

bool begin_asset_drag_source(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    bool directory,
    const char* label);

bool accept_asset_drag_drop(editor_asset_payload& payload);

} // namespace arc::editor
