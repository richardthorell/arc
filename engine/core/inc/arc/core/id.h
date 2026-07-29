#pragma once

#include <array>
#include <charconv>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace arc::core
{

/// @brief Text layouts supported for UUID serialization.
enum class uuid_text_format
{
    /// Thirty-two lowercase hexadecimal digits.
    compact,
    /// RFC-4122-style groups separated by hyphens.
    hyphenated
};

/// @brief A type-safe 128-bit universally unique identifier.
/// @tparam Tag Empty tag type that prevents IDs from unrelated domains from mixing.
template <class Tag> struct uuid
{
    /// Most-significant 64 serialized bits.
    std::uint64_t high{};
    /// Least-significant 64 serialized bits.
    std::uint64_t low{};

    /// @brief Return whether this identifier is not the reserved null value.
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return high != 0 || low != 0;
    }

    friend constexpr auto operator<=>(const uuid&, const uuid&) noexcept = default;
};

/// @brief Hash a tagged UUID without erasing its type.
/// @tparam Tag UUID domain tag.
template <class Tag> struct uuid_hash
{
    /// @brief Return a stable in-process hash for an identifier.
    [[nodiscard]] constexpr std::size_t operator()(uuid<Tag> value) const noexcept
    {
        const auto mixed = value.high ^ (value.low + 0x9e3779b97f4a7c15ull + (value.high << 6u) + (value.high >> 2u));
        return static_cast<std::size_t>(mixed);
    }
};

/// @brief Format a tagged UUID as lowercase hexadecimal text.
/// @tparam Tag UUID domain tag.
/// @param value Identifier to format.
/// @param format Compact or hyphenated output layout.
/// @return Canonical text for the requested layout.
template <class Tag>
[[nodiscard]] inline std::string to_string(uuid<Tag> value, uuid_text_format format = uuid_text_format::compact)
{
    constexpr char digits[] = "0123456789abcdef";
    const bool hyphenated = format == uuid_text_format::hyphenated;
    std::string output;
    output.reserve(hyphenated ? 36 : 32);
    for (std::size_t byte_index = 0; byte_index < 16; ++byte_index)
    {
        if (hyphenated && (byte_index == 4 || byte_index == 6 || byte_index == 8 || byte_index == 10))
            output.push_back('-');
        const std::uint64_t half = byte_index < 8 ? value.high : value.low;
        const auto half_index = byte_index < 8 ? byte_index : byte_index - 8;
        const auto byte = static_cast<std::uint8_t>(half >> ((7u - half_index) * 8u));
        output.push_back(digits[byte >> 4u]);
        output.push_back(digits[byte & 0x0fu]);
    }
    return output;
}

/// @brief Parse a tagged UUID from compact or hyphenated hexadecimal text.
/// @tparam Tag UUID domain tag.
/// @param text Canonical 32-character compact or 36-character hyphenated representation.
/// @return Parsed non-null identifier, or an empty optional for malformed/null input.
template <class Tag> [[nodiscard]] inline std::optional<uuid<Tag>> parse_uuid(std::string_view text) noexcept
{
    std::array<char, 32> compact{};
    if (text.size() == 32)
    {
        for (std::size_t index = 0; index < text.size(); ++index)
            compact[index] = text[index];
    }
    else if (text.size() == 36 && text[8] == '-' && text[13] == '-' && text[18] == '-' && text[23] == '-')
    {
        std::size_t output{};
        for (char character : text)
            if (character != '-') compact[output++] = character;
    }
    else
        return std::nullopt;

    uuid<Tag> value;
    const auto high = std::from_chars(compact.data(), compact.data() + 16, value.high, 16);
    const auto low = std::from_chars(compact.data() + 16, compact.data() + 32, value.low, 16);
    if (high.ec != std::errc{} || low.ec != std::errc{} || high.ptr != compact.data() + 16 ||
        low.ptr != compact.data() + 32 || !value.valid())
        return std::nullopt;
    return value;
}

/// @brief A type-safe scalar identifier with a reserved invalid value.
/// @tparam Tag Empty tag type that distinguishes unrelated identifier domains.
/// @tparam Rep Unsigned integral storage type.
/// @tparam Invalid Reserved invalid representation.
template <class Tag, class Rep = std::uint64_t, Rep Invalid = std::numeric_limits<Rep>::max()>
    requires(std::is_integral_v<Rep> && std::is_unsigned_v<Rep>)
struct strong_id
{
    /// Unsigned serialized representation type.
    using representation_type = Rep;
    /// Reserved representation used for invalid handles.
    static constexpr Rep invalid_value = Invalid;

    /// Serialized representation.
    Rep value{Invalid};

    /// @brief Return whether this identifier is not the reserved invalid value.
    [[nodiscard]] constexpr bool valid() const noexcept
    {
        return value != Invalid;
    }

    /// @brief Return the underlying serialized representation.
    [[nodiscard]] constexpr Rep representation() const noexcept
    {
        return value;
    }

    friend constexpr auto operator<=>(const strong_id&, const strong_id&) noexcept = default;
};

/// @brief Hash a tagged scalar identifier without erasing its type.
/// @tparam Tag Identifier domain tag.
/// @tparam Rep Identifier representation type.
/// @tparam Invalid Reserved invalid representation.
template <class Tag, class Rep, Rep Invalid> struct strong_id_hash
{
    /// @brief Return a stable in-process hash for an identifier.
    [[nodiscard]] constexpr std::size_t operator()(strong_id<Tag, Rep, Invalid> value) const noexcept
    {
        return static_cast<std::size_t>(value.representation());
    }
};

static_assert(sizeof(uuid<struct uuid_layout_test_tag>) == 16);

} // namespace arc::core
