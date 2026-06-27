#include <geometric/point.h>
#include <geometric/line.h>
#include <geometric/box.h>
#include <geometric/circle.h>
#include <geometric/geometric.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

TEST_CASE("points are strong math-vector backed types")
{
    using namespace arc::geometric;
    using arc::math::vector3f;

    static_assert(std::is_same_v<point3f, point<float, 3>>);

    point3f p{ 1.0f, 2.0f, 3.0f };
    vector3f v{ 10.0f, 20.0f, 30.0f };

    const point3f moved = p + v;
    REQUIRE(moved[0] == 11.0f);
    REQUIRE(moved[2] == 33.0f);

    const point3f restored = moved - v;
    REQUIRE(restored[1] == 2.0f);

    const vector3f delta = moved - p;
    REQUIRE(delta[0] == 10.0f);
    REQUIRE(distance(p, moved) == Catch::Approx(37.41657f));
}

TEST_CASE("lines rays and segments can be sampled")
{
    using namespace arc::geometric;

    line3f infinite{ point3f{ 1.0f, 0.0f, 0.0f }, arc::math::vector3f{ 0.0f, 2.0f, 0.0f } };
    ray3f half_line{ point3f{ 1.0f, 0.0f, 0.0f }, arc::math::vector3f{ 0.0f, 2.0f, 0.0f } };
    segment3f finite{ point3f{ 0.0f, 0.0f, 0.0f }, point3f{ 10.0f, 0.0f, 0.0f } };

    REQUIRE(point_at(infinite, 3.0f)[1] == 6.0f);
    REQUIRE(point_at(half_line, 2.0f)[1] == 4.0f);
    REQUIRE(point_at(finite, 0.25f)[0] == 2.5f);
}

TEST_CASE("boxes normalize and support containment intersection and expansion")
{
    using namespace arc::geometric;

    box3f bounds{ point3f{ 10.0f, 10.0f, 10.0f }, point3f{ 0.0f, 0.0f, 0.0f } };
    REQUIRE(bounds.min[0] == 0.0f);
    REQUIRE(bounds.max[2] == 10.0f);

    REQUIRE(contains(bounds, point3f{ 5.0f, 5.0f, 5.0f }));
    REQUIRE_FALSE(contains(bounds, point3f{ 11.0f, 5.0f, 5.0f }));

    box3f overlapping{ point3f{ 9.0f, 9.0f, 9.0f }, point3f{ 20.0f, 20.0f, 20.0f } };
    box3f separate{ point3f{ 11.0f, 11.0f, 11.0f }, point3f{ 20.0f, 20.0f, 20.0f } };
    REQUIRE(intersects(bounds, overlapping));
    REQUIRE_FALSE(intersects(bounds, separate));

    const point3f clamped = closest_point(bounds, point3f{ -5.0f, 3.0f, 15.0f });
    REQUIRE(clamped[0] == 0.0f);
    REQUIRE(clamped[1] == 3.0f);
    REQUIRE(clamped[2] == 10.0f);

    const point3f c = center(bounds);
    const auto e = extents(bounds);
    REQUIRE(c[0] == 5.0f);
    REQUIRE(e[1] == 5.0f);

    const box3f grown = expand(bounds, point3f{ -2.0f, 3.0f, 12.0f });
    REQUIRE(grown.min[0] == -2.0f);
    REQUIRE(grown.max[2] == 12.0f);

    const box3f padded = expand(bounds, 1.0f);
    REQUIRE(padded.min[0] == -1.0f);
    REQUIRE(padded.max[1] == 11.0f);
}

TEST_CASE("circles and spheres support containment intersection and closest point")
{
    using namespace arc::geometric;

    circlef c{ point2f{ 0.0f, 0.0f }, 5.0f };
    circlef other{ point2f{ 8.0f, 0.0f }, 4.0f };
    circlef far{ point2f{ 20.0f, 0.0f }, 4.0f };

    REQUIRE(contains(c, point2f{ 3.0f, 4.0f }));
    REQUIRE_FALSE(contains(c, point2f{ 6.0f, 0.0f }));
    REQUIRE(intersects(c, other));
    REQUIRE_FALSE(intersects(c, far));

    const point2f circle_edge = closest_point(c, point2f{ 10.0f, 0.0f });
    REQUIRE(circle_edge[0] == Catch::Approx(5.0f));
    REQUIRE(circle_edge[1] == Catch::Approx(0.0f));

    spheref s{ point3f{ 0.0f, 0.0f, 0.0f }, 2.0f };
    spheref touching{ point3f{ 4.0f, 0.0f, 0.0f }, 2.0f };
    REQUIRE(contains(s, point3f{ 1.0f, 0.0f, 0.0f }));
    REQUIRE(intersects(s, touching));

    const point3f sphere_edge = closest_point(s, point3f{ 0.0f, 0.0f, 5.0f });
    REQUIRE(sphere_edge[2] == Catch::Approx(2.0f));
}
