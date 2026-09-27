#include <arc/navigation/navigation.h>

#include <cassert>
#include <limits>

int main()
{
    using namespace arc::navigation;

    assert(validate(agent_definition{}) == validation_error::none);

    auto invalid_agent = agent_definition{};
    invalid_agent.radius = 0.0F;
    assert(validate(invalid_agent) == validation_error::invalid_agent);

    invalid_agent = agent_definition{};
    invalid_agent.max_slope_degrees = 90.0F;
    assert(validate(invalid_agent) == validation_error::invalid_agent);

    navigation_bounds bounds{};
    bounds.minimum = {-10.0F, -2.0F, -10.0F};
    bounds.maximum = {10.0F, 4.0F, 10.0F};
    assert(validate(bounds) == validation_error::none);

    bounds.minimum[0] = 11.0F;
    assert(validate(bounds) == validation_error::invalid_bounds);

    path_query query{};
    query.start = {0.0F, 0.0F, 0.0F};
    query.goal = {10.0F, 0.0F, 10.0F};
    assert(validate(query) == validation_error::none);

    query.goal[1] = std::numeric_limits<float>::quiet_NaN();
    assert(validate(query) == validation_error::invalid_query);

    query = path_query{};
    query.max_path_length = 0.0F;
    assert(validate(query) == validation_error::invalid_query);

    path_result result{};
    assert(result.status == path_status::unreachable);
    assert(result.points.empty());

    static_assert(tile_coordinate{2, 4} == tile_coordinate{2, 4});
    static_assert(!(tile_coordinate{2, 4} == tile_coordinate{4, 2}));

    return 0;
}
