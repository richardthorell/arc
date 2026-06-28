#include <arc/editor/editor_viewport.h>

#include <algorithm>

namespace arc::editor
{

void editor_viewport::set_size(float width, float height) noexcept
{
    width_ = static_cast<std::uint32_t>(std::max(0.0f, width));
    height_ = static_cast<std::uint32_t>(std::max(0.0f, height));
}

void editor_viewport::set_focused(bool value) noexcept
{
    focused_ = value;
}

void editor_viewport::set_hovered(bool value) noexcept
{
    hovered_ = value;
}

std::uint32_t editor_viewport::width() const noexcept
{
    return width_;
}

std::uint32_t editor_viewport::height() const noexcept
{
    return height_;
}

bool editor_viewport::valid() const noexcept
{
    return width_ > 0 && height_ > 0;
}

bool editor_viewport::focused() const noexcept
{
    return focused_;
}

bool editor_viewport::hovered() const noexcept
{
    return hovered_;
}

} // namespace arc::editor
