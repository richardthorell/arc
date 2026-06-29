#include <arc/editor/panels/console_panel.h>

#include <arc/editor/ui/theme.h>
#include <arc/editor/ui/widgets.h>

#include <imgui.h>

#include <array>
#include <chrono>
#include <ctime>
#include <string>
#include <string_view>

namespace arc::editor
{
namespace
{

std::string_view display_level(log_level level)
{
    switch (level)
    {
    case log_level::trace:
        return "Trace";
    case log_level::debug:
        return "Debug";
    case log_level::info:
        return "Info";
    case log_level::warn:
        return "Warn";
    case log_level::error:
        return "Error";
    case log_level::fatal:
        return "Fatal";
    }
    return "Log";
}

ImVec4 color_for_level(log_level level)
{
    switch (level)
    {
    case log_level::trace:
        return ImVec4(0.62f, 0.67f, 0.72f, 1.0f);
    case log_level::debug:
        return ImVec4(0.39f, 0.67f, 1.0f, 1.0f);
    case log_level::info:
        return ImVec4(0.45f, 0.75f, 1.0f, 1.0f);
    case log_level::warn:
        return ImVec4(1.0f, 0.78f, 0.28f, 1.0f);
    case log_level::error:
    case log_level::fatal:
        return ImVec4(1.0f, 0.32f, 0.32f, 1.0f);
    }
    return ui::colors::text;
}

std::string format_timestamp(std::chrono::system_clock::time_point timestamp)
{
    if (timestamp == std::chrono::system_clock::time_point{})
        timestamp = std::chrono::system_clock::now();

    const std::time_t value = std::chrono::system_clock::to_time_t(timestamp);
    std::tm local_time{};
#if defined(_WIN32)
    localtime_s(&local_time, &value);
#else
    localtime_r(&value, &local_time);
#endif

    char buffer[16]{};
    std::strftime(buffer, sizeof(buffer), "%H:%M:%S", &local_time);
    return buffer;
}

bool level_visible(const editor_ui_state& state, log_level level)
{
    switch (level)
    {
    case log_level::trace:
        return state.console_show_trace;
    case log_level::debug:
        return state.console_show_debug;
    case log_level::info:
        return state.console_show_info;
    case log_level::warn:
        return state.console_show_warn;
    case log_level::error:
    case log_level::fatal:
        return state.console_show_error;
    }
    return true;
}

bool text_matches_filter(const console_log_entry& entry, std::string_view filter)
{
    if (filter.empty())
        return true;
    return entry.message.find(filter) != std::string::npos ||
        entry.category.find(filter) != std::string::npos ||
        display_level(entry.level).find(filter) != std::string_view::npos;
}

void draw_level_pill(log_level level)
{
    const std::string_view text = display_level(level);
    const ImVec4 color = color_for_level(level);
    const ImVec2 min = ImGui::GetCursorScreenPos();
    const ImVec2 text_size = ImGui::CalcTextSize(text.data());
    const ImVec2 max(min.x + text_size.x + ui::scaled(16.0f), min.y + text_size.y + ui::scaled(5.0f));
    ImGui::GetWindowDrawList()->AddRectFilled(min, max, ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 0.85f)), ui::scaled(4.0f));
    ImGui::GetWindowDrawList()->AddText(ImVec2(min.x + ui::scaled(8.0f), min.y + ui::scaled(2.0f)), IM_COL32(5, 7, 9, 255), text.data());
    ImGui::Dummy(ImVec2(max.x - min.x, max.y - min.y));
}

void copy_menu(const console_log_entry& entry, const std::string& timestamp)
{
    if (ImGui::BeginPopupContextItem("log-row-menu"))
    {
        if (ImGui::MenuItem("Copy Message"))
            ImGui::SetClipboardText(entry.message.c_str());
        if (ImGui::MenuItem("Copy Category"))
            ImGui::SetClipboardText(entry.category.c_str());
        if (ImGui::MenuItem("Copy Full Entry"))
        {
            const std::string full =
                "[" + timestamp + "] [" + std::string(display_level(entry.level)) + "] " +
                entry.category + " " + entry.message;
            ImGui::SetClipboardText(full.c_str());
        }
        ImGui::EndPopup();
    }
}

} // namespace

void draw_console_panel(editor_console_sink& sink, editor_ui_state& state)
{
    if (!ui::begin_panel("Console"))
    {
        ui::end_panel();
        return;
    }

    const auto entries = sink.entries();
    std::array<int, 4> counts{};
    for (const auto& entry : entries)
    {
        switch (entry.level)
        {
        case log_level::trace:
        case log_level::debug:
            ++counts[0];
            break;
        case log_level::info:
            ++counts[1];
            break;
        case log_level::warn:
            ++counts[2];
            break;
        case log_level::error:
        case log_level::fatal:
            ++counts[3];
            break;
        }
    }

    ui::panel_toolbar_begin();
    if (ImGui::Button("Clear"))
        sink.clear();
    ImGui::SameLine();
    ui::toggle_chip("Collapse", &state.console_collapse);
    ImGui::SameLine();
    ui::toggle_chip("Auto-scroll", &state.console_auto_scroll);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
    ui::search_box("##console-filter", "Search logs...", state.console_filter, sizeof(state.console_filter));
    ui::panel_toolbar_end();

    ui::log_level_chip("Trace/Debug", &state.console_show_debug, counts[0], color_for_level(log_level::debug));
    ImGui::SameLine();
    ui::log_level_chip("Info", &state.console_show_info, counts[1], color_for_level(log_level::info));
    ImGui::SameLine();
    ui::log_level_chip("Warn", &state.console_show_warn, counts[2], color_for_level(log_level::warn));
    ImGui::SameLine();
    ui::log_level_chip("Error", &state.console_show_error, counts[3], color_for_level(log_level::error));
    state.console_show_trace = state.console_show_debug;
    ImGui::Separator();

    ImGui::BeginChild("console-scroll", ImVec2(0.0f, -ImGui::GetFrameHeightWithSpacing() - ui::scaled(4.0f)), true);
    std::string previous_key;
    int visible_index = 0;
    for (const auto& entry : entries)
    {
        if (!level_visible(state, entry.level) || !text_matches_filter(entry, state.console_filter))
            continue;

        const std::string key = std::string(display_level(entry.level)) + entry.category + entry.message;
        if (state.console_collapse && key == previous_key)
            continue;
        previous_key = key;

        const std::string time = format_timestamp(entry.timestamp);
        ImGui::PushID(visible_index++);
        const ImVec2 row_min = ImGui::GetCursorScreenPos();
        if ((visible_index % 2) == 0)
        {
            ImGui::GetWindowDrawList()->AddRectFilled(
                row_min,
                ImVec2(row_min.x + ImGui::GetContentRegionAvail().x, row_min.y + ui::scaled(26.0f)),
                IM_COL32(255, 255, 255, 8));
        }

        ImGui::TextDisabled("[%s]", time.c_str());
        ImGui::SameLine(ui::scaled(86.0f));
        draw_level_pill(entry.level);
        ImGui::SameLine(ui::scaled(156.0f));
        if (!entry.category.empty())
        {
            ImGui::PushStyleColor(ImGuiCol_Text, ui::colors::text_muted);
            ImGui::TextUnformatted(entry.category.c_str());
            ImGui::PopStyleColor();
            ImGui::SameLine(ui::scaled(260.0f));
        }

        ImGui::PushStyleColor(ImGuiCol_Text, color_for_level(entry.level));
        ImGui::TextWrapped("%s", entry.message.c_str());
        ImGui::PopStyleColor();
        copy_menu(entry, time);
        ImGui::PopID();
    }

    if (state.console_auto_scroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - ui::scaled(4.0f))
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();

    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::InputTextWithHint("##console-command", "> Enter console command...", state.console_command, sizeof(state.console_command), ImGuiInputTextFlags_EnterReturnsTrue))
    {
        if (state.console_command[0] != '\0')
        {
            arc::info("console", state.console_command);
            state.console_command[0] = '\0';
        }
    }

    ui::end_panel();
}

} // namespace arc::editor
