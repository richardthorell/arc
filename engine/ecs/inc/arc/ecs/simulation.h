#pragma once

#include <arc/memory/memory.h>

#include <cstdint>
#include <compare>
#include <limits>
#include <span>

namespace arc::ecs
{

struct simulation_tick_id
{
    std::uint64_t value{};

    constexpr bool valid() const noexcept { return value != 0; }
    friend constexpr bool operator==(simulation_tick_id, simulation_tick_id) noexcept = default;
    friend constexpr auto operator<=>(simulation_tick_id, simulation_tick_id) noexcept = default;
};

enum class runtime_world_role : std::uint8_t
{
    server,
    client,
    editor_preview
};

struct runtime_service_id
{
    std::uint64_t value{};

    constexpr bool valid() const noexcept { return value != 0; }
    friend constexpr bool operator==(runtime_service_id, runtime_service_id) noexcept = default;
};

class runtime_service_provider
{
public:
    virtual ~runtime_service_provider() = default;
    virtual void* find_service(runtime_service_id id) noexcept = 0;
    virtual const void* find_service(runtime_service_id id) const noexcept = 0;
};

struct random_stream_id
{
    std::uint64_t value{};

    constexpr bool valid() const noexcept { return value != 0; }
    friend constexpr bool operator==(random_stream_id, random_stream_id) noexcept = default;
};

enum class simulation_input_kind : std::uint8_t
{
    key,
    mouse_button,
    mouse_position,
    mouse_wheel,
    focus
};

enum class simulation_input_action : std::uint8_t
{
    pressed,
    released,
    changed
};

struct simulation_input_command
{
    simulation_input_kind kind{ simulation_input_kind::key };
    simulation_input_action action{ simulation_input_action::changed };
    std::int32_t code{};
    std::uint32_t modifiers{};
    std::int32_t x{};
    std::int32_t y{};
    float value{};
    bool repeat{};
};

struct simulation_input_snapshot
{
    std::uint64_t revision{};
    std::span<const simulation_input_command> commands;
};

constexpr std::uint64_t stable_hash_64(const char* text) noexcept
{
    std::uint64_t value = 1469598103934665603ull;
    while (*text)
    {
        value ^= static_cast<std::uint8_t>(*text++);
        value *= 1099511628211ull;
    }
    return value;
}

constexpr random_stream_id make_random_stream_id(const char* name) noexcept
{
    return { stable_hash_64(name) };
}

/** ARC-owned PCG32 stream. Its output is stable across standard-library implementations. */
class random_stream
{
public:
    constexpr random_stream() noexcept = default;
    constexpr random_stream(std::uint64_t state, std::uint64_t sequence) noexcept
    {
        seed(state, sequence);
    }

    constexpr void seed(std::uint64_t state, std::uint64_t sequence) noexcept
    {
        state_ = 0;
        increment_ = (sequence << 1u) | 1u;
        next_u32();
        state_ += state;
        next_u32();
    }

    constexpr std::uint32_t next_u32() noexcept
    {
        const std::uint64_t old_state = state_;
        state_ = old_state * 6364136223846793005ull + increment_;
        const auto shifted = static_cast<std::uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
        const auto rotation = static_cast<std::uint32_t>(old_state >> 59u);
        return (shifted >> rotation) | (shifted << ((-rotation) & 31u));
    }

    constexpr float next_float() noexcept
    {
        return static_cast<float>(next_u32() >> 8u) * (1.0f / 16777216.0f);
    }

    constexpr std::uint32_t range(std::uint32_t upper_exclusive) noexcept
    {
        if (upper_exclusive == 0)
            return 0;
        const std::uint32_t threshold = static_cast<std::uint32_t>(-upper_exclusive) % upper_exclusive;
        for (;;)
        {
            const std::uint32_t value = next_u32();
            if (value >= threshold)
                return value % upper_exclusive;
        }
    }

private:
    std::uint64_t state_{ 0x853c49e6748fea9bull };
    std::uint64_t increment_{ 0xda3e39cb94b95bdbull };
};

constexpr std::uint64_t splitmix64(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31u);
}

inline random_stream make_random_stream(
    std::uint64_t process_seed,
    std::uint64_t world_id,
    simulation_tick_id tick,
    random_stream_id stream,
    std::uint64_t stable_subject = 0) noexcept
{
    std::uint64_t state = splitmix64(process_seed);
    state = splitmix64(state ^ world_id);
    state = splitmix64(state ^ tick.value);
    state = splitmix64(state ^ stream.value);
    const std::uint64_t sequence = splitmix64(state ^ stable_subject);
    return random_stream(state, sequence);
}

struct system_execution_info
{
    simulation_tick_id tick{};
    runtime_world_role world_role{ runtime_world_role::client };
    std::uint64_t world_id{};
    std::uint64_t process_seed{};
    float delta_seconds{};
    float fixed_delta_seconds{};
    float frame_delta_seconds{};
    float interpolation_alpha{};
    bool presentation{};
    const simulation_input_snapshot* input{};
    runtime_service_provider* services{};
    tick_arena* tick_memory{};
    frame_arena* frame_memory{};
};

} // namespace arc::ecs
