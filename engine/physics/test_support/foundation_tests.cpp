#include <arc/physics/physics.h>

#include <cassert>
#include <limits>

int main()
{
    using namespace arc::physics;

    body_desc body{};
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
