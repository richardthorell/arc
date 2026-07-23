#pragma once

#include <cstdint>

namespace arc::editor
{

/**
 * @brief Editor viewport state shared between UI and future renderer integration.
 */
class editor_viewport
{
public:
    /**
     * @brief Set the latest available viewport size in pixels.
     */
    void set_size(float width, float height) noexcept;

    /**
     * @brief Set the viewport rectangle in screen/window coordinates.
     */
    void set_screen_rect(float x, float y, float width, float height) noexcept;

    /**
     * @brief Set whether the viewport currently has keyboard focus.
     */
    void set_focused(bool value) noexcept;

    /**
     * @brief Set whether the pointer is currently over the viewport.
     */
    void set_hovered(bool value) noexcept;

    /**
     * @brief Return requested render width in pixels.
     */
    std::uint32_t width() const noexcept;

    /**
     * @brief Return requested render height in pixels.
     */
    std::uint32_t height() const noexcept;

    /**
     * @brief Return whether the viewport has a positive render size.
     */
    bool valid() const noexcept;

    /**
     * @brief Return whether the viewport has keyboard focus.
     */
    bool focused() const noexcept;

    /**
     * @brief Return whether the pointer is over the viewport.
     */
    bool hovered() const noexcept;

    /**
     * @brief Return viewport origin X in screen/window coordinates.
     */
    float screen_x() const noexcept;

    /**
     * @brief Return viewport origin Y in screen/window coordinates.
     */
    float screen_y() const noexcept;

    /**
     * @brief Return whether a screen/window coordinate lies within the viewport.
     */
    bool contains_screen_point(float x, float y) const noexcept;

    /**
     * @brief Convert screen/window X coordinate to viewport-local X coordinate.
     */
    float local_x(float screen_x) const noexcept;

    /**
     * @brief Convert screen/window Y coordinate to viewport-local Y coordinate.
     */
    float local_y(float screen_y) const noexcept;

private:
    std::uint32_t width_{};
    std::uint32_t height_{};
    float screen_x_{};
    float screen_y_{};
    bool focused_{};
    bool hovered_{};
};

} // namespace arc::editor
