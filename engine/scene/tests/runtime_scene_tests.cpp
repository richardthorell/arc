#include <arc/scene/runtime_scene.h>

#include <arc/ecs/hierarchy.h>
#include <arc/persistence/persistence.h>
#include <arc/scene/components.h>
#include <arc/scene/persistence.h>

#include <catch2/catch_test_macros.hpp>

#include <span>
#include <string>
#include <vector>

namespace
{

arc::persistence::archive_document authored_scene(bool with_child)
{
    namespace persistence = arc::persistence;
    persistence::component_persistence_registry registry;
    REQUIRE(arc::scene::register_persistence_components(registry));

    persistence::archive_document document;
    document.id = {1, 2};
    document.name = "Packaged Startup";

    persistence::archive_entity_record parent;
    parent.id = {3, 4};
    const arc::scene::name_component name{"Root Entity"};
    persistence::archive_component_record encoded;
    REQUIRE(registry.encode(arc::ecs::component_metadata<arc::scene::name_component>().id, &name, encoded));
    parent.components.push_back(std::move(encoded));
    document.entities.push_back(std::move(parent));

    if (with_child)
    {
        persistence::archive_entity_record child;
        child.id = {5, 6};
        child.parent = {3, 4};
        child.sibling_order = 0;
        document.entities.push_back(std::move(child));
    }
    return document;
}

std::vector<std::byte> cooked_bytes(const arc::persistence::archive_document& document)
{
    const auto result = arc::persistence::write_tagged_binary(document, "windows-x64-vulkan12");
    REQUIRE(result);
    return result.value();
}

} // namespace

TEST_CASE("D1.7 cooked scene materializes persisted entity IDs, components and hierarchy")
{
    const auto data = cooked_bytes(authored_scene(true));
    arc::ecs::world world;
    CHECK(arc::scene::materialize_cooked_scene(world, data).empty());
    REQUIRE(world.live_count() == 2);

    arc::ecs::entity root;
    arc::ecs::entity child;
    for (auto entity : world.entities())
    {
        const auto id = world.get<arc::ecs::persistent_id_component>(entity).value;
        if (id == arc::ecs::entity_guid{3, 4}) root = entity;
        if (id == arc::ecs::entity_guid{5, 6}) child = entity;
    }
    REQUIRE(root.valid());
    REQUIRE(child.valid());
    CHECK(world.get<arc::scene::name_component>(root).value == "Root Entity");
    REQUIRE(world.has<arc::ecs::hierarchy_component>(child));
    CHECK(world.get<arc::ecs::hierarchy_component>(child).parent == root);
}

TEST_CASE("D1.7 cooked scene refuses damaged archive without creating ECS entities")
{
    auto data = cooked_bytes(authored_scene(false));
    REQUIRE_FALSE(data.empty());
    data.back() ^= std::byte{1};
    arc::ecs::world world;
    CHECK_FALSE(arc::scene::materialize_cooked_scene(world, data).empty());
    CHECK(world.live_count() == 0);
}

TEST_CASE("D1.7 cooked scene rolls back partially materialized worlds")
{
    auto document = authored_scene(true);
    arc::persistence::archive_component_record unsupported;
    unsupported.type = {0x11111111, 0x22222222};
    unsupported.name = "Unregistered Gameplay Component";
    document.entities.back().components.push_back(std::move(unsupported));

    const auto data = cooked_bytes(document);
    arc::ecs::world world;
    const auto error = arc::scene::materialize_cooked_scene(world, data);
    CHECK(error.find("Unregistered Gameplay Component") != std::string::npos);
    CHECK(world.live_count() == 0);
}
