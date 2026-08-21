#include <arc/editor/arc_host.h>
#include <arc/render/renderer.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <memory>

namespace
{
nlohmann::json viewport_state(const std::shared_ptr<arc::editor::arc_host>& host, std::string viewport_id,
                              std::uint64_t request_id)
{
    const auto response = host->query(arc::editor::host_query_envelope{
        .request_id = request_id,
        .payload = arc::editor::host_viewport_state_query{.viewport_id = std::move(viewport_id)},
    });
    REQUIRE(response.succeeded);
    const auto payload = nlohmann::json::parse(response.payload_json, nullptr, false);
    REQUIRE_FALSE(payload.is_discarded());
    return payload;
}
} // namespace

TEST_CASE("viewport surfaces keep dimensions and render options independent")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    REQUIRE(host->execute(arc::editor::host_viewport_resize_command{
        .viewport_id = "viewport-1", .width = 1024, .height = 768}).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_create_command{
        .viewport_id = "material-preview", .width = 320, .height = 240}).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_set_render_options_command{
        .viewport_id = "material-preview", .grid = false, .camera_speed = 9.0f}).succeeded);

    const auto scene = viewport_state(host, "viewport-1", 1);
    const auto preview = viewport_state(host, "material-preview", 2);

    CHECK(scene.at("viewportId") == "viewport-1");
    CHECK(scene.at("width") == 1024);
    CHECK(scene.at("height") == 768);
    CHECK(scene.at("renderOptions").at("grid") == true);
    CHECK(scene.at("renderOptions").at("cameraSpeed") == 4.0f);

    CHECK(preview.at("viewportId") == "material-preview");
    CHECK(preview.at("width") == 320);
    CHECK(preview.at("height") == 240);
    CHECK(preview.at("renderOptions").at("grid") == false);
    CHECK(preview.at("renderOptions").at("cameraSpeed") == 9.0f);

    const auto missing = host->execute(arc::editor::host_viewport_resize_command{
        .viewport_id = "missing", .width = 10, .height = 10});
    CHECK_FALSE(missing.succeeded);
    CHECK(missing.error == "Viewport is not attached");
}

TEST_CASE("viewport surfaces keep local frame indices independent")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    REQUIRE(host->execute(arc::editor::host_viewport_create_command{
        .viewport_id = "shader-preview", .width = 256, .height = 256}).succeeded);

    const auto scene_frame = host->request_viewport(
        {.viewport_id = "viewport-1", .frame_index = 7, .width = 640, .height = 480});
    const auto preview_frame = host->request_viewport(
        {.viewport_id = "shader-preview", .frame_index = 3, .width = 256, .height = 256});
    CHECK_FALSE(scene_frame.submitted);
    CHECK_FALSE(preview_frame.submitted);

    const auto scene = viewport_state(host, "viewport-1", 3);
    const auto preview = viewport_state(host, "shader-preview", 4);
    CHECK(scene.at("frameIndex") == 7);
    CHECK(preview.at("frameIndex") == 3);
    CHECK(scene.at("submitted") == false);
    CHECK(preview.at("submitted") == false);
}

TEST_CASE("detaching one viewport surface does not detach another")
{
    auto renderer = std::make_unique<arc::render::renderer>();
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::move(renderer));

    REQUIRE(host->execute(arc::editor::host_viewport_create_command{
        .viewport_id = "material-preview", .width = 300, .height = 300}).succeeded);
    REQUIRE(host->execute(arc::editor::host_viewport_detach_command{.viewport_id = "material-preview"}).succeeded);

    const auto scene = viewport_state(host, "viewport-1", 5);
    const auto preview = viewport_state(host, "material-preview", 6);
    CHECK(scene.at("viewportId") == "viewport-1");
    CHECK(preview.at("viewportId") == "material-preview");
    CHECK(preview.at("width") == 0);
    CHECK(preview.at("height") == 0);
    CHECK(preview.at("submitted") == false);
}
