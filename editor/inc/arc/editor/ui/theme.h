#pragma once

#include <imgui.h>

namespace arc::editor::ui
{

inline constexpr float ui_scale = 1.28f;

float scaled(float value) noexcept;

namespace colors
{
inline constexpr ImVec4 shell_bg{ 0.035f, 0.041f, 0.047f, 1.0f };
inline constexpr ImVec4 panel_bg{ 0.060f, 0.068f, 0.078f, 1.0f };
inline constexpr ImVec4 panel_bg_alt{ 0.075f, 0.085f, 0.096f, 1.0f };
inline constexpr ImVec4 child_bg{ 0.046f, 0.052f, 0.060f, 1.0f };
inline constexpr ImVec4 border{ 0.145f, 0.165f, 0.185f, 1.0f };
inline constexpr ImVec4 border_subtle{ 0.105f, 0.120f, 0.135f, 1.0f };
inline constexpr ImVec4 text{ 0.835f, 0.865f, 0.895f, 1.0f };
inline constexpr ImVec4 text_muted{ 0.500f, 0.545f, 0.590f, 1.0f };
inline constexpr ImVec4 accent{ 0.225f, 0.520f, 0.920f, 1.0f };
inline constexpr ImVec4 accent_hover{ 0.315f, 0.620f, 1.000f, 1.0f };
inline constexpr ImVec4 danger{ 0.950f, 0.275f, 0.275f, 1.0f };
inline constexpr ImVec4 warning{ 0.980f, 0.720f, 0.250f, 1.0f };
inline constexpr ImVec4 success{ 0.300f, 0.720f, 0.380f, 1.0f };
} // namespace colors

void apply_dark_theme();

} // namespace arc::editor::ui
