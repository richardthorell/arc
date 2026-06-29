#include <arc/editor/editor_viewport.h>

#include <algorithm>

namespace arc::editor
{

void editor_viewport::set_size(float width, float height) noexcept
{
    width_ = static_cast<std::uint32_t>(std::max(0.0f, width));
    height_ = static_cast<std::uint32_t>(std::max(0.0f, height));
}

void editor_viewport::set_screen_rect(float x, float y, float width, float height) noexcept
{
    screen_x_ = x;
    screen_y_ = y;
    set_size(width, height);
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

float editor_viewport::screen_x() const noexcept
{
    return screen_x_;
}

float editor_viewport::screen_y() const noexcept
{
    return screen_y_;
}

bool editor_viewport::contains_screen_point(float x, float y) const noexcept
{
    return x >= screen_x_ &&
        y >= screen_y_ &&
        x < screen_x_ + static_cast<float>(width_) &&
        y < screen_y_ + static_cast<float>(height_);
}

float editor_viewport::local_x(float screen_x) const noexcept
{
    return screen_x - screen_x_;
}

float editor_viewport::local_y(float screen_y) const noexcept
{
    return screen_y - screen_y_;
}

} // namespace arc::editor
