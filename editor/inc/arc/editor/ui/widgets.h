#pragma once

#include <cstddef>

#include <imgui.h>

namespace arc::editor::ui
{

bool toolbar_button(const char* label, bool active = false, ImVec2 size = ImVec2(0.0f, 0.0f));
void toolbar_separator();
bool search_box(const char* id, const char* hint, char* buffer, std::size_t buffer_size);
bool begin_panel(const char* name, bool* open = nullptr, ImGuiWindowFlags flags = 0);
void end_panel();
bool section_header(const char* label, bool default_open = true);
bool component_card_begin(const char* label, bool default_open = true);
void component_card_end();
bool entity_row(const char* label, const char* icon, bool selected, bool visible = true, bool locked = false);
bool toggle_chip(const char* label, bool* value, int count = -1);
void log_level_chip(const char* label, bool* value, int count, ImVec4 color);
bool vec3_field(const char* label, float values[3], float speed = 0.01f, float min_value = 0.0f, float max_value = 0.0f);
bool float_field(const char* label, float* value, float speed = 0.01f);
bool color_field3(const char* label, float values[3]);
bool checkbox_field(const char* label, bool* value);
void empty_state(const char* title, const char* message);
void muted_text(const char* text);
void panel_toolbar_begin();
void panel_toolbar_end();

} // namespace arc::editor::ui
