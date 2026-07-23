#include <arc/ecs/identity.h>

#include <array>
#include <charconv>
#include <chrono>
#include <random>

namespace arc::ecs
{

entity_guid generate_entity_guid() noexcept
{
    static thread_local std::mt19937_64 generator([] {
        std::random_device random;
        const auto clock = static_cast<std::uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::seed_seq seed{
            random(), random(),
            static_cast<std::uint32_t>(clock),
            static_cast<std::uint32_t>(clock >> 32u) };
        return std::mt19937_64(seed);
    }());

    entity_guid result{ generator(), generator() };
    result.high = (result.high & 0xffffffffffff0fffull) | 0x0000000000004000ull;
    result.low = (result.low & 0x3fffffffffffffffull) | 0x8000000000000000ull;
    return result;
}

std::string to_string(entity_guid value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 36> output{};
    const std::array<std::uint8_t, 16> bytes{
        static_cast<std::uint8_t>(value.high >> 56u), static_cast<std::uint8_t>(value.high >> 48u),
        static_cast<std::uint8_t>(value.high >> 40u), static_cast<std::uint8_t>(value.high >> 32u),
        static_cast<std::uint8_t>(value.high >> 24u), static_cast<std::uint8_t>(value.high >> 16u),
        static_cast<std::uint8_t>(value.high >> 8u), static_cast<std::uint8_t>(value.high),
        static_cast<std::uint8_t>(value.low >> 56u), static_cast<std::uint8_t>(value.low >> 48u),
        static_cast<std::uint8_t>(value.low >> 40u), static_cast<std::uint8_t>(value.low >> 32u),
        static_cast<std::uint8_t>(value.low >> 24u), static_cast<std::uint8_t>(value.low >> 16u),
        static_cast<std::uint8_t>(value.low >> 8u), static_cast<std::uint8_t>(value.low) };
    std::size_t cursor{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        if (index == 4 || index == 6 || index == 8 || index == 10)
            output[cursor++] = '-';
        output[cursor++] = digits[bytes[index] >> 4u];
        output[cursor++] = digits[bytes[index] & 0x0fu];
    }
    return { output.data(), output.size() };
}

std::optional<entity_guid> parse_entity_guid(std::string_view value) noexcept
{
    if (value.size() != 36 || value[8] != '-' || value[13] != '-' || value[18] != '-' || value[23] != '-')
        return std::nullopt;
    std::array<char, 32> compact{};
    std::size_t cursor{};
    for (char character : value)
        if (character != '-')
            compact[cursor++] = character;
    entity_guid result{};
    const auto high = std::from_chars(compact.data(), compact.data() + 16, result.high, 16);
    const auto low = std::from_chars(compact.data() + 16, compact.data() + 32, result.low, 16);
    if (high.ec != std::errc{} || high.ptr != compact.data() + 16 ||
        low.ec != std::errc{} || low.ptr != compact.data() + 32 || !result.valid())
        return std::nullopt;
    return result;
}

} // namespace arc::ecs
