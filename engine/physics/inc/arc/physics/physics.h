#pragma once

#include <arc/math/quaternion.h>
#include <arc/math/vector.h>

#include <cstdint>
#include <limits>
#include <variant>

namespace arc::physics
{

template <typename Tag>
class handle
{
public:
    using value_type = std::uint64_t;

    constexpr handle() noexcept = default;
    explicit constexpr handle(value_type value) noexcept : value_(value) {}

    [[nodiscard]] constexpr value_type value() const noexcept { return value_; }
    [[nodiscard]] constexpr bool valid() const noexcept { return value_ != 0; }
    explicit constexpr operator bool() const noexcept { return valid(); }

    friend constexpr bool operator==(handle, handle) noexcept = default;

private:
    value_type value_ = 0;
};

struct world_handle_tag;
struct body_handle_tag;
struct shape_handle_tag;
struct material_handle_tag;

using world_handle = handle<world_handle_tag>;
using body_handle = handle<body_handle_tag>;
using shape_handle = handle<shape_handle_tag>;
using material_handle = handle<material_handle_tag>;

// Keep the existing body identifier name as the public compatibility spelling while
// the backend-neutral handle contract grows around it.
using body_id = body_handle;
inline constexpr body_id invalid_body_id{};

enum class body_motion : std::uint8_t
{
    static_body,
    kinematic,
    dynamic
};

struct sphere_shape
{
    float radius = 0.5F;
};

struct box_shape
{
    math::vector<float, 3> half_extents{0.5F, 0.5F, 0.5F};
};

struct capsule_shape
{
    float radius = 0.5F;
    float half_height = 0.5F;
};

using collision_shape = std::variant<sphere_shape, box_shape, capsule_shape>;

struct body_definition
{
    body_motion motion = body_motion::static_body;
    collision_shape shape = box_shape{};
    math::vector<float, 3> position{};
    math::quaternion<float> rotation{};
    float mass = 1.0F;
    float linear_damping = 0.0F;
    float angular_damping = 0.0F;
    bool enable_gravity = true;
};

struct ray_query
{
    math::vector<float, 3> origin{};
    math::vector<float, 3> direction{0.0F, 0.0F, 1.0F};
    float max_distance = std::numeric_limits<float>::infinity();
    std::uint32_t layer_mask = ~std::uint32_t{0};
};

struct ray_hit
{
    body_id body = invalid_body_id;
    math::vector<float, 3> position{};
    math::vector<float, 3> normal{};
    float distance = 0.0F;
};

enum class validation_error : std::uint8_t
{
    none,
    invalid_shape,
    invalid_mass,
    invalid_damping,
    invalid_query
};

[[nodiscard]] validation_error validate(const body_definition& definition) noexcept;
[[nodiscard]] validation_error validate(const ray_query& query) noexcept;

} // namespace arc::physics
