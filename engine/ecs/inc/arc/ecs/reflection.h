#pragma once

#include <array>
#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace arc::ecs
{

struct component_type_id
{
    std::uint64_t high{};
    std::uint64_t low{};

    constexpr bool valid() const noexcept { return high != 0 || low != 0; }
    friend constexpr auto operator<=>(component_type_id, component_type_id) noexcept = default;
};

struct component_type_id_hash
{
    std::size_t operator()(component_type_id value) const noexcept
    {
        return static_cast<std::size_t>(value.high ^ (value.low + 0x9e3779b97f4a7c15ull +
            (value.high << 6u) + (value.high >> 2u)));
    }
};

inline std::string to_string(component_type_id value)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(32, '0');
    for (std::size_t index = 0; index < 16; ++index)
    {
        const std::uint64_t source = index < 8 ? value.high : value.low;
        const std::size_t byte_index = index < 8 ? index : index - 8;
        const auto byte = static_cast<std::uint8_t>(source >> ((7 - byte_index) * 8u));
        result[index * 2] = digits[byte >> 4u];
        result[index * 2 + 1] = digits[byte & 0x0fu];
    }
    return result;
}

inline std::optional<component_type_id> parse_component_type_id(std::string_view value) noexcept
{
    if (value.size() != 32)
        return std::nullopt;
    component_type_id result{};
    const auto high = std::from_chars(value.data(), value.data() + 16, result.high, 16);
    const auto low = std::from_chars(value.data() + 16, value.data() + 32, result.low, 16);
    if (high.ec != std::errc{} || low.ec != std::errc{} ||
        high.ptr != value.data() + 16 || low.ptr != value.data() + 32 || !result.valid())
        return std::nullopt;
    return result;
}

using component_field_id = std::uint64_t;
using runtime_component_index = std::uint16_t;
inline constexpr runtime_component_index invalid_runtime_component_index =
    static_cast<runtime_component_index>(-1);

enum class reflected_field_kind : std::uint8_t
{
    unknown,
    boolean,
    signed_integer,
    unsigned_integer,
    floating_point,
    string,
    enumeration,
    vector2,
    vector3,
    vector4,
    quaternion,
    matrix,
    entity_reference,
    asset_reference,
    structure,
    sequence
};

enum class reflected_field_flags : std::uint16_t
{
    none = 0,
    serialized = 1u << 0u,
    replicated = 1u << 1u,
    editable = 1u << 2u,
    transient = 1u << 3u,
    prefab_override = 1u << 4u
};

constexpr reflected_field_flags operator|(reflected_field_flags lhs, reflected_field_flags rhs) noexcept
{
    return static_cast<reflected_field_flags>(
        static_cast<std::uint16_t>(lhs) | static_cast<std::uint16_t>(rhs));
}

constexpr bool has_flag(reflected_field_flags value, reflected_field_flags flag) noexcept
{
    return (static_cast<std::uint16_t>(value) & static_cast<std::uint16_t>(flag)) != 0;
}

struct component_field_descriptor
{
    component_field_id id{};
    std::string_view name;
    std::string_view display_name;
    reflected_field_kind kind{ reflected_field_kind::unknown };
    reflected_field_flags flags{
        reflected_field_flags::serialized |
        reflected_field_flags::editable |
        reflected_field_flags::prefab_override };
};

struct component_descriptor
{
    component_type_id id{};
    std::string_view canonical_name;
    std::string_view display_name;
    std::uint32_t schema_version{ 1 };
    std::size_t size{};
    std::size_t alignment{};
    std::span<const component_field_descriptor> fields;
    bool custom_serialization{};
    bool custom_replication{};
};

template <class T>
constexpr const component_descriptor& component_metadata() noexcept;

/**
 * Registration is performed during world/application setup and then frozen.
 * Sorting by the schema-owned 128-bit ID makes compact indices deterministic
 * regardless of static initialization or module load order.
 */
class component_type_registry
{
public:
    bool register_component(const component_descriptor& descriptor)
    {
        if (frozen_ || !descriptor.id.valid() ||
            std::any_of(descriptors_.begin(), descriptors_.end(), [&](const component_descriptor* existing) {
                return existing->id == descriptor.id;
            }))
            return false;
        descriptors_.push_back(&descriptor);
        return true;
    }

    template <class T>
    bool register_component()
    {
        return register_component(component_metadata<T>());
    }

    bool freeze()
    {
        if (frozen_)
            return true;
        if (descriptors_.size() >= invalid_runtime_component_index)
            return false;
        std::sort(descriptors_.begin(), descriptors_.end(), [](const auto* lhs, const auto* rhs) {
            return lhs->id < rhs->id;
        });
        frozen_ = true;
        return true;
    }

    bool frozen() const noexcept { return frozen_; }
    std::size_t size() const noexcept { return descriptors_.size(); }

    runtime_component_index runtime_index(component_type_id id) const noexcept
    {
        if (!frozen_)
            return invalid_runtime_component_index;
        const auto found = std::lower_bound(descriptors_.begin(), descriptors_.end(), id,
            [](const component_descriptor* descriptor, component_type_id value) {
                return descriptor->id < value;
            });
        return found != descriptors_.end() && (*found)->id == id
            ? static_cast<runtime_component_index>(found - descriptors_.begin())
            : invalid_runtime_component_index;
    }

    const component_descriptor* descriptor(runtime_component_index index) const noexcept
    {
        return frozen_ && index < descriptors_.size() ? descriptors_[index] : nullptr;
    }

private:
    std::vector<const component_descriptor*> descriptors_;
    bool frozen_{};
};

namespace detail
{

constexpr std::uint64_t fnv1a(std::string_view value, std::uint64_t seed) noexcept
{
    std::uint64_t result = seed;
    for (const char character : value)
    {
        result ^= static_cast<std::uint8_t>(character);
        result *= 1099511628211ull;
    }
    return result;
}

template <class T>
constexpr std::string_view compiler_type_name() noexcept
{
#if defined(_MSC_VER)
    constexpr std::string_view signature = __FUNCSIG__;
#else
    constexpr std::string_view signature = __PRETTY_FUNCTION__;
#endif
    return signature;
}

constexpr component_type_id fallback_type_id(std::string_view name) noexcept
{
    return {
        fnv1a(name, 14695981039346656037ull),
        fnv1a(name, 1099511628211ull)
    };
}

} // namespace detail

/**
 * Generated component metadata specializes this trait. Unregistered local/test
 * components remain usable, but their compiler-derived identity is not a
 * persistence contract.
 */
template <class T>
struct component_traits
{
    static constexpr bool reflected = false;
    static constexpr auto canonical_name = detail::compiler_type_name<T>();
    static constexpr component_type_id id = detail::fallback_type_id(canonical_name);
    static constexpr std::array<component_field_descriptor, 0> fields{};

    static constexpr component_descriptor descriptor{
        id, canonical_name, canonical_name, 1, sizeof(T), alignof(T), fields, false, false
    };
};

template <class T>
constexpr component_type_id component_type() noexcept
{
    return component_traits<std::remove_cv_t<T>>::id;
}

template <class T>
constexpr const component_descriptor& component_metadata() noexcept
{
    return component_traits<std::remove_cv_t<T>>::descriptor;
}

inline std::size_t field_index(const component_descriptor& descriptor, component_field_id id) noexcept
{
    for (std::size_t index = 0; index < descriptor.fields.size(); ++index)
        if (descriptor.fields[index].id == id)
            return index;
    return static_cast<std::size_t>(-1);
}

} // namespace arc::ecs
