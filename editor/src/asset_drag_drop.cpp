#include <arc/editor/asset_drag_drop.h>

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

namespace arc::editor
{
namespace
{

struct imgui_asset_payload
{
    int kind{};
    char relative_path[260]{};
};

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

const char* payload_id_for_kind(editor_asset_kind kind) noexcept
{
    switch (kind)
    {
    case editor_asset_kind::material:
        return material_asset_payload_id;
    case editor_asset_kind::texture:
    case editor_asset_kind::environment:
        return texture_asset_payload_id;
    default:
        return asset_path_payload_id;
    }
}

bool decode_payload(const ImGuiPayload* payload, editor_asset_payload& out)
{
    if (!payload || payload->DataSize != static_cast<int>(sizeof(imgui_asset_payload)))
        return false;

    const auto* data = static_cast<const imgui_asset_payload*>(payload->Data);
    out.kind = static_cast<editor_asset_kind>(data->kind);
    out.relative_path = std::filesystem::path{ data->relative_path };
    return !out.relative_path.empty();
}

bool accept_payload_type(const char* id, editor_asset_payload& out)
{
    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(id))
        return decode_payload(payload, out);
    return false;
}

} // namespace

editor_asset_kind classify_asset_path(const std::filesystem::path& path, bool directory)
{
    if (directory)
        return editor_asset_kind::directory;

    const std::string ext = lowercase(path.extension().string());
    if (ext == ".arcmat")
        return editor_asset_kind::material;
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" || ext == ".dds")
        return editor_asset_kind::texture;
    if (ext == ".hdr")
        return editor_asset_kind::environment;
    if (ext == ".glb" || ext == ".gltf" || ext == ".fbx")
        return editor_asset_kind::scene;
    if (ext == ".vert" || ext == ".frag" || ext == ".spv")
        return editor_asset_kind::shader;
    if (ext == ".scene")
        return editor_asset_kind::scene;
    return editor_asset_kind::unknown;
}

bool is_texture_asset_kind(editor_asset_kind kind) noexcept
{
    return kind == editor_asset_kind::texture || kind == editor_asset_kind::environment;
}

std::filesystem::path asset_relative_path(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path)
{
    std::error_code ec;
    const auto relative = std::filesystem::relative(path, asset_root, ec);
    if (!ec && !relative.empty())
        return relative.lexically_normal();
    return path.filename();
}

bool make_asset_payload(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    bool directory,
    editor_asset_payload& payload)
{
    payload.kind = classify_asset_path(path, directory);
    payload.relative_path = asset_relative_path(asset_root, path).generic_string();
    return payload.kind != editor_asset_kind::directory && !payload.relative_path.empty();
}

bool begin_asset_drag_source(
    const std::filesystem::path& asset_root,
    const std::filesystem::path& path,
    bool directory,
    const char* label)
{
    editor_asset_payload payload;
    if (!make_asset_payload(asset_root, path, directory, payload))
        return false;

    if (!ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
        return false;

    imgui_asset_payload data{};
    data.kind = static_cast<int>(payload.kind);
    const auto relative = payload.relative_path.generic_string();
    std::snprintf(data.relative_path, sizeof(data.relative_path), "%s", relative.c_str());
    ImGui::SetDragDropPayload(payload_id_for_kind(payload.kind), &data, sizeof(data));
    ImGui::TextUnformatted(label && label[0] != '\0' ? label : relative.c_str());
    ImGui::EndDragDropSource();
    return true;
}

bool accept_asset_drag_drop(editor_asset_payload& payload)
{
    if (!ImGui::BeginDragDropTarget())
        return false;

    bool accepted = accept_payload_type(material_asset_payload_id, payload);
    if (!accepted)
        accepted = accept_payload_type(texture_asset_payload_id, payload);
    if (!accepted)
        accepted = accept_payload_type(asset_path_payload_id, payload);

    ImGui::EndDragDropTarget();
    return accepted;
}

} // namespace arc::editor
