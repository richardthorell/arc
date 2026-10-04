#include <arc/physics/backend.h>
#include <arc/physics/physics.h>

#include <cassert>
#include <limits>
#include <type_traits>

namespace
{

class test_backend final : public arc::physics::backend
{
public:
    [[nodiscard]] std::string_view name() const noexcept override
    {
        return "test";
    }

    [[nodiscard]] bool initialized() const noexcept override
    {
        return true;
    }
};

arc::physics::backend_ptr create_test_backend(const arc::physics::backend_config&)
{
    return std::make_unique<test_backend>();
}

} // namespace

int main()
{
    using namespace arc::physics;

    static_assert(!std::is_same_v<world_handle, body_handle>);
    static_assert(!std::is_same_v<body_handle, shape_handle>);
    static_assert(!std::is_same_v<shape_handle, material_handle>);
    static_assert(std::is_abstract_v<backend>);
    static_assert(std::is_same_v<decltype(&create_test_backend), backend_factory>);

    constexpr world_handle world{7};
    static_assert(world.valid());
    static_assert(world.value() == 7);
    static_assert(!world_handle{}.valid());
    static_assert(body_id{} == invalid_body_id);

    backend_config config{};
    backend_ptr backend_instance = create_test_backend(config);
    assert(backend_instance);
    assert(backend_instance->initialized());
    assert(backend_instance->name() == "test");

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
