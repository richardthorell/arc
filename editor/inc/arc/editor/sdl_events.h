#pragma once

#include <arc/event.h>

#include <SDL3/SDL_events.h>

namespace arc::editor
{

/**
 * @brief Translate an SDL event into an ARC event when a matching event exists.
 */
bool translate_sdl_event(const SDL_Event& source, event& destination) noexcept;

} // namespace arc::editor
