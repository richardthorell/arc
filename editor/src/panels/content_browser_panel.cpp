#include <arc/editor/panels/content_browser_panel.h>

#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>
#include <arc/diagnostics/diagnostics.h>

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <vector>

namespace arc::editor
{
namespace
{

struct asset_entry
{
    std::filesystem::path path;
    bool directory{};
};

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool matches_filter(const std::filesystem::path& path, const char* filter)
{
    if (!filter || filter[0] == '\0')
        return true;
    return lowercase(path.filename().string()).find(lowercase(filter)) != std::string::npos;
}

const char* asset_type_label(const asset_entry& entry)
{
    if (entry.directory)
        return "Folder";

    const std::string ext = lowercase(entry.path.extension().string());
    if (ext == ".glb" || ext == ".gltf")
        return "Mesh";
    if (ext == ".vert" || ext == ".frag" || ext == ".spv")
        return "Shader";
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga")
        return "Texture";
    if (ext == ".scene")
        return "Scene";
    return "File";
}

ImVec4 asset_color(const asset_entry& entry)
{
    if (entry.directory)
        return ImVec4(0.50f, 0.38f, 0.18f, 1.0f);
    const std::string type = asset_type_label(entry);
    if (type == "Mesh")
        return ImVec4(0.33f, 0.43f, 0.55f, 1.0f);
    if (type == "Shader")
        return ImVec4(0.28f, 0.48f, 0.40f, 1.0f);
    if (type == "Texture")
        return ImVec4(0.34f, 0.36f, 0.50f, 1.0f);
    if (type == "Scene")
        return ImVec4(0.42f, 0.34f, 0.52f, 1.0f);
    return ImVec4(0.30f, 0.33f, 0.36f, 1.0f);
}

std::vector<asset_entry> scan_directory(const std::filesystem::path& directory)
{
    std::vector<asset_entry> entries;
    std::error_code ec;
    for (const auto& item : std::filesystem::directory_iterator(directory, ec))
        entries.push_back({ item.path(), item.is_directory(ec) });

    std::sort(entries.begin(), entries.end(), [](const asset_entry& lhs, const asset_entry& rhs) {
        if (lhs.directory != rhs.directory)
            return lhs.directory > rhs.directory;
        return lowercase(lhs.path.filename().string()) < lowercase(rhs.path.filename().string());
    });
    return entries;
}

void log_placeholder_action(const char* action, const std::filesystem::path& path)
{
    arc::info("editor.assets", std::string(action) + " requested for " + path.filename().string());
}

bool draw_asset_tile(const asset_entry& entry, bool selected)
{
    const ImVec2 tile_size(ui::scaled(118.0f), ui::scaled(104.0f));
    const ImVec2 start = ImGui::GetCursorScreenPos();
    ImGui::PushID(entry.path.string().c_str());
    const bool clicked = ImGui::InvisibleButton("asset-tile", tile_size);
    const bool hovered = ImGui::IsItemHovered();
    const bool double_clicked = hovered && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);

    ImDrawList* draw_list = ImGui::GetWindowDrawList();
    const ImVec4 color = asset_color(entry);
    const ImU32 bg = ImGui::ColorConvertFloat4ToU32(hovered || selected ? ui::colors::panel_bg_alt : ui::colors::child_bg);
    draw_list->AddRectFilled(start, ImVec2(start.x + tile_size.x, start.y + tile_size.y), bg, ui::scaled(5.0f));
    if (selected)
        draw_list->AddRect(start, ImVec2(start.x + tile_size.x, start.y + tile_size.y), ImGui::ColorConvertFloat4ToU32(ui::colors::accent), ui::scaled(5.0f), 0, ui::scaled(1.5f));

    const ImVec2 swatch_min(start.x + ui::scaled(12.0f), start.y + ui::scaled(12.0f));
    const ImVec2 swatch_max(start.x + tile_size.x - ui::scaled(12.0f), start.y + ui::scaled(56.0f));
    draw_list->AddRectFilled(swatch_min, swatch_max, ImGui::ColorConvertFloat4ToU32(color), ui::scaled(5.0f));
    draw_list->AddText(ImVec2(swatch_min.x + ui::scaled(8.0f), swatch_min.y + ui::scaled(12.0f)), IM_COL32(235, 240, 244, 255), asset_type_label(entry));

    ImGui::SetCursorScreenPos(ImVec2(start.x + ui::scaled(10.0f), start.y + ui::scaled(64.0f)));
    ImGui::PushTextWrapPos(start.x + tile_size.x - ui::scaled(8.0f));
    ImGui::TextUnformatted(entry.path.filename().string().c_str());
    ImGui::PopTextWrapPos();

    if (ImGui::BeginPopupContextItem("asset-menu"))
    {
        if (ImGui::MenuItem("Open"))
            log_placeholder_action("Open", entry.path);
        if (ImGui::MenuItem("Rename"))
            log_placeholder_action("Rename", entry.path);
        if (ImGui::MenuItem("Delete"))
            log_placeholder_action("Delete", entry.path);
        if (ImGui::MenuItem("Show in Explorer/Finder"))
            log_placeholder_action("Reveal", entry.path);
        ImGui::EndPopup();
    }

    ImGui::PopID();
    return clicked || double_clicked;
}

} // namespace

void draw_content_browser_panel(const editor_asset_state& editor_assets)
{
    if (!ui::begin_panel("Content Browser"))
    {
        ui::end_panel();
        return;
    }

    static std::filesystem::path remembered_root;
    static std::filesystem::path current_directory;
    static std::filesystem::path selected_path;
    static char asset_search[128]{};

    if (remembered_root != editor_assets.root)
    {
        remembered_root = editor_assets.root;
        current_directory = editor_assets.root;
        selected_path.clear();
    }
    if (current_directory.empty())
        current_directory = editor_assets.root;
    if (!std::filesystem::exists(current_directory))
        current_directory = editor_assets.root;

    ui::panel_toolbar_begin();
    if (ImGui::Button("Back") && current_directory != editor_assets.root)
        current_directory = current_directory.parent_path();
    ImGui::SameLine();
    std::error_code ec;
    const auto relative = std::filesystem::relative(current_directory, editor_assets.root, ec);
    ui::muted_text(ec ? "Assets" : ("Assets / " + relative.generic_string()).c_str());
    ImGui::SameLine(ImGui::GetWindowWidth() - ui::scaled(220.0f));
    ImGui::SetNextItemWidth(ui::scaled(200.0f));
    ui::search_box("##asset-search", "Search assets...", asset_search, sizeof(asset_search));
    ui::panel_toolbar_end();

    if (!std::filesystem::exists(editor_assets.root))
    {
        ui::empty_state("Assets folder missing", "Create an assets directory to browse project content.");
        ui::end_panel();
        return;
    }

    const auto entries = scan_directory(current_directory);
    const float tile_width = ui::scaled(130.0f);
    const int columns = std::max(1, static_cast<int>(ImGui::GetContentRegionAvail().x / tile_width));
    if (ImGui::BeginTable("asset-grid", columns, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_PadOuterX))
    {
        int column = 0;
        for (const auto& entry : entries)
        {
            if (!matches_filter(entry.path, asset_search))
                continue;

            ImGui::TableNextColumn();
            const bool selected = entry.path == selected_path;
            if (draw_asset_tile(entry, selected))
            {
                selected_path = entry.path;
                if (entry.directory && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
                    current_directory = entry.path;
            }
            ++column;
        }

        if (column == 0)
            ui::empty_state("No assets found", "Try a different search or add files to this folder.");
        ImGui::EndTable();
    }

    ui::end_panel();
}

} // namespace arc::editor
