#include <arc/ecs/identity.h>

#include <chrono>
#include <random>

namespace arc::ecs
{

entity_guid generate_entity_guid() noexcept
{
    static thread_local std::mt19937_64 generator(
        []
        {
            std::random_device random;
            const auto clock =
                static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
            std::seed_seq seed{random(), random(), static_cast<std::uint32_t>(clock),
                               static_cast<std::uint32_t>(clock >> 32u)};
            return std::mt19937_64(seed);
        }());

    entity_guid result{generator(), generator()};
    result.high = (result.high & 0xffffffffffff0fffull) | 0x0000000000004000ull;
    result.low = (result.low & 0x3fffffffffffffffull) | 0x8000000000000000ull;
    return result;
}

std::string to_string(entity_guid value)
{
    return core::to_string(value, core::uuid_text_format::hyphenated);
}

std::optional<entity_guid> parse_entity_guid(std::string_view value) noexcept
{
    return core::parse_uuid<entity_guid_tag>(value);
}

} // namespace arc::ecs
