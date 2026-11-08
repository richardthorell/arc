#pragma once

#include <cstdint>

namespace arc::platform::android
{

// Host-owned state only. No ARC runtime or Vulkan objects live in this class.
// In particular, losing a window does not mean the process is destroyed.
enum class lifecycle_event
{
    start,
    resume,
    pause,
    stop,
    gain_focus,
    lose_focus,
    window_created,
    window_destroyed,
    window_resized,
    config_changed,
    low_memory,
    destroy
};

struct lifecycle_snapshot
{
    bool started{};
    bool resumed{};
    bool focused{};
    bool window_available{};
    bool destroy_requested{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint64_t window_generation{};
};

// All state transitions occur on the android_main/app glue callback thread.
// Repeated commands are idempotent; generations change on window attachment.
class lifecycle_state
{
public:
    void apply(lifecycle_event event, std::uint32_t width = 0, std::uint32_t height = 0) noexcept;
    [[nodiscard]] const lifecycle_snapshot& snapshot() const noexcept { return state_; }
    [[nodiscard]] bool can_present() const noexcept
    {
        return state_.started && state_.resumed && state_.focused && state_.window_available &&
               !state_.destroy_requested && state_.width > 0 && state_.height > 0;
    }

private:
    lifecycle_snapshot state_;
};

} // namespace arc::platform::android
