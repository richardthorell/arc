#include <arc/editor/arc_host.h>
#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

TEST_CASE("agent selection uses the native editor selection state and event")
{
    arc::editor::arc_host_manager manager;
    auto host = manager.acquire(std::make_unique<arc::render::renderer>());
    REQUIRE(host->open_project({.name = "Agent Selection", .root = std::filesystem::temp_directory_path()}, {}).succeeded);

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::plane})
                .succeeded);
    const auto first_snapshot = host->selected_entity_snapshot();
    const auto first = first_snapshot.entity;
    const auto first_guid = first_snapshot.guid;
    REQUIRE(first.valid());
    REQUIRE_FALSE(first_guid.empty());

    REQUIRE(host->execute(arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube})
                .succeeded);
    REQUIRE(host->selected_entity_snapshot().entity != first);
    (void)host->poll_events();

    REQUIRE(host->execute(arc::editor::host_select_entity_command{.entity = first}).succeeded);
    const auto selected = host->selected_entity_snapshot();
    REQUIRE(selected.entity == first);
    REQUIRE(selected.guid == first_guid);
    REQUIRE(selected.selection_count == 1);
    REQUIRE(selected.selected_guids == std::vector<std::string>{first_guid});

    const auto selected_query = host->query(arc::editor::host_query_envelope{
        .request_id = 1, .payload = arc::editor::host_selected_entity_query{}});
    REQUIRE(selected_query.succeeded);
    const auto selected_json = nlohmann::json::parse(selected_query.payload_json);
    REQUIRE(selected_json.at("selectionCount") == 1);
    REQUIRE(selected_json.at("selectedGuids").at(0) == first_guid);

    const auto select_events = host->poll_events();
    REQUIRE(std::any_of(select_events.begin(), select_events.end(), [&](const auto& event)
                        {
                            return event.event_type == arc::editor::host_event_type::entity_selected &&
                                   event.entity == first;
                        }));

    REQUIRE(host->execute(arc::editor::host_clear_selection_command{}).succeeded);
    const auto cleared = host->selected_entity_snapshot();
    REQUIRE_FALSE(cleared.entity.valid());
    REQUIRE(cleared.selection_count == 0);
    REQUIRE(cleared.selected_guids.empty());

    const auto clear_events = host->poll_events();
    REQUIRE(std::any_of(clear_events.begin(), clear_events.end(), [](const auto& event)
                        { return event.event_type == arc::editor::host_event_type::entity_selected; }));
}
