#include <arc/physics/physics.h>

#include <cassert>
#include <limits>
#include <type_traits>

int main()
{
    using namespace arc::physics;

    static_assert(!std::is_same_v<world_handle, body_handle>);
    static_assert(!std::is_same_v<body_handle, shape_handle>);
    static_assert(!std::is_same_v<shape_handle, material_handle>);

    constexpr world_handle world{7};
    static_assert(world.valid());
    static_assert(world.value() == 7);
    static_assert(!world_handle{}.valid());
    static_assert(body_id{} == invalid_body_id);

    body_definition body{};
    assert(validate(body) == validation_error::none);

    body.motion = body_motion::dynamic;
    body.mass = 0.0F;
    assert(validate(body) == validation_error::invalid_mass);

    body.mass = 1.0F;
    body.shape = sphere_shape{0.0F};
    assert(validate(body) == validation_error::invalid_shape);

    body.shape = capsule_shape{0.5F, 1.0F};
    body.linear_damping = -0.1F;
    assert(validate(body) == validation_error::invalid_damping);

    ray_query query{};
    assert(validate(query) == validation_error::none);

    query.direction = arc::math::vector<float, 3>{0.0F, 0.0F, 0.0F};
    assert(validate(query) == validation_error::invalid_query);

    query.direction = arc::math::vector<float, 3>{0.0F, 0.0F, 1.0F};
    query.max_distance = std::numeric_limits<float>::quiet_NaN();
    assert(validate(query) == validation_error::invalid_query);

    return 0;
}
