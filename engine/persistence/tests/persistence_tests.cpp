#include <arc/persistence/persistence.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace
{

struct test_value_component
{
    double value{};
};

constexpr arc::ecs::component_field_descriptor test_fields[] = {
    { 0x1001, "value", "Value", arc::ecs::reflected_field_kind::floating_point,
        arc::ecs::reflected_field_flags::serialized |
            arc::ecs::reflected_field_flags::editable,
        offsetof(test_value_component, value), sizeof(double) },
};
constexpr arc::ecs::component_descriptor test_component{
    { 0x1020304050607080ull, 0x90a0b0c0d0e0f001ull },
    "arc.test_component",
    "Test",
    2,
    sizeof(test_value_component),
    alignof(test_value_component),
    test_fields,
    false,
    false
};

arc::persistence::component_persistence_registry make_registry()
{
    arc::persistence::component_persistence_registry registry;
    REQUIRE(registry.register_component({ &test_component, { "Test" } }));
    REQUIRE(registry.freeze());
    return registry;
}

arc::persistence::archive_document make_document()
{
    arc::persistence::archive_document document;
    document.id = { 1, 2 };
    document.name = "Persistence fixture";
    arc::persistence::archive_entity_record entity;
    entity.id = { 3, 4 };
    arc::persistence::archive_component_record component;
    component.type = test_component.id;
    component.name = "Test";
    component.schema_version = 2;
    arc::persistence::archive_value value;
    value.kind = arc::persistence::archive_value_kind::floating_point;
    value.floating_point = 42.5;
    component.fields.push_back({ test_fields[0].id, "value", value, true });
    arc::persistence::archive_value unknown;
    unknown.kind = arc::persistence::archive_value_kind::string;
    unknown.string = "preserve me";
    component.fields.push_back({ 0xf000000000000001ull, "futureField", unknown, false });
    entity.components.push_back(std::move(component));
    document.entities.push_back(std::move(entity));
    return document;
}

class temporary_directory
{
public:
    temporary_directory()
    {
        path = std::filesystem::temp_directory_path() /
            ("arc-persistence-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
    }
    ~temporary_directory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    std::filesystem::path path;
};

} // namespace

TEST_CASE("reflected JSON preserves unknown fields and verifies integrity")
{
    const auto document = make_document();
    const auto json_result = arc::persistence::write_reflected_json(document, true);
    REQUIRE(json_result);
    const auto& json = json_result.value();
    REQUIRE(arc::persistence::verify_json_document(json));
    const auto registry = make_registry();
    const auto loaded = arc::persistence::read_reflected_json(json, registry);
    REQUIRE(loaded.succeeded());
    REQUIRE(loaded.integrity_verified);
    const auto& fields = loaded.document.entities[0].components[0].fields;
    const auto unknown = std::find_if(fields.begin(), fields.end(),
        [](const auto& field) { return field.name == "futureField"; });
    REQUIRE(unknown != fields.end());
    REQUIRE_FALSE(unknown->known);
    REQUIRE(unknown->value.string == "preserve me");
}

TEST_CASE("ordinary reflected components use generated field access codecs")
{
    const auto registry = make_registry();
    const test_value_component source{ 19.75 };
    arc::persistence::archive_component_record encoded;
    REQUIRE(registry.encode(test_component.id, &source, encoded));
    REQUIRE(encoded.fields.size() == 1);
    REQUIRE(encoded.fields[0].value.floating_point == Catch::Approx(19.75));

    test_value_component destination;
    REQUIRE(registry.decode(test_component.id, &destination, encoded));
    REQUIRE(destination.value == Catch::Approx(source.value));
}

TEST_CASE("tagged binary is deterministic and rejects component corruption")
{
    const auto document = make_document();
    const auto first_result = arc::persistence::write_tagged_binary(
        document, "windows-x64-vulkan12");
    const auto second_result = arc::persistence::write_tagged_binary(
        document, "windows-x64-vulkan12");
    REQUIRE(first_result);
    REQUIRE(second_result);
    const auto& first = first_result.value();
    REQUIRE(first == second_result.value());
    const auto registry = make_registry();
    const auto loaded = arc::persistence::read_tagged_binary(first, registry);
    REQUIRE(loaded.succeeded());
    REQUIRE(loaded.target_identity == "windows-x64-vulkan12");
    REQUIRE(loaded.document == document);

    auto corrupt = first;
    corrupt.back() ^= std::byte{ 0x40 };
    REQUIRE_FALSE(arc::persistence::read_tagged_binary(corrupt, registry).succeeded());
}

TEST_CASE("migration registry requires consecutive edges")
{
    arc::persistence::schema_migration_registry migrations;
    REQUIRE_FALSE(migrations.register_component(test_component.id, 1, 3,
        [](auto&) { return arc::persistence::persistence_status::success(); }));
    REQUIRE(migrations.freeze());
}

TEST_CASE("migration registry applies consecutive component and document migrations")
{
    arc::persistence::schema_migration_registry migrations;
    REQUIRE(migrations.register_component(test_component.id, 1, 2,
        [](auto& component) {
            component.name = "Migrated Test";
            return arc::persistence::persistence_status::success();
        }));
    REQUIRE(migrations.register_document(
        arc::persistence::document_kind::scene, 1, 2,
        [](auto& document) {
            document.name += " v2";
            return arc::persistence::persistence_status::success();
        }));
    REQUIRE(migrations.register_document(
        arc::persistence::document_kind::scene, 2, 3,
        [](auto& document) {
            document.name += " v3";
            return arc::persistence::persistence_status::success();
        }));
    REQUIRE(migrations.freeze());

    auto component = make_document().entities[0].components[0];
    component.schema_version = 1;
    REQUIRE(migrations.migrate(component, 2));
    REQUIRE(component.schema_version == 2);
    REQUIRE(component.name == "Migrated Test");

    auto document = make_document();
    document.format_version = 1;
    REQUIRE(migrations.migrate(document, 3));
    REQUIRE(document.format_version == 3);
    REQUIRE(document.name.ends_with("v2 v3"));
}

TEST_CASE("migration registry rejects gaps within a registered chain")
{
    arc::persistence::schema_migration_registry migrations;
    REQUIRE(migrations.register_component(test_component.id, 1, 2,
        [](auto&) { return arc::persistence::persistence_status::success(); }));
    REQUIRE(migrations.register_component(test_component.id, 3, 4,
        [](auto&) { return arc::persistence::persistence_status::success(); }));
    const auto frozen = migrations.freeze();
    REQUIRE_FALSE(frozen);
    REQUIRE(frozen.error().message.find("gap") != std::string::npos);
}

TEST_CASE("archives reject unresolved parents without mutating a document")
{
    auto document = make_document();
    document.entities[0].parent = { 99, 100 };
    const auto text_result = arc::persistence::write_reflected_json(document, false);
    REQUIRE(text_result);
    const auto registry = make_registry();
    const auto loaded = arc::persistence::read_reflected_json(text_result.value(), registry);
    REQUIRE_FALSE(loaded.succeeded());
    REQUIRE(loaded.error.find("parent") != std::string::npos);
}

TEST_CASE("document store rotates backups and recovers the newest valid generation")
{
    temporary_directory directory;
    const auto destination = directory.path / "scene.arcscene";
    arc::persistence::document_store store;
    auto document = make_document();
    for (int generation = 0; generation < 4; ++generation)
    {
        document.name = "generation-" + std::to_string(generation);
        const auto text = arc::persistence::write_reflected_json(document, true);
        REQUIRE(text);
        REQUIRE(store.save_json(destination, text.value()).succeeded);
    }
    REQUIRE(std::filesystem::exists(destination.string() + ".bak1"));
    REQUIRE(std::filesystem::exists(destination.string() + ".bak2"));
    REQUIRE(std::filesystem::exists(destination.string() + ".bak3"));

    {
        std::ofstream corrupt(destination, std::ios::binary | std::ios::trunc);
        corrupt << "{broken";
    }
    const auto recovered = store.load_json(destination);
    REQUIRE(recovered.succeeded);
    REQUIRE(recovered.recovered);
    REQUIRE(recovered.source_path == std::filesystem::path(destination.string() + ".bak1"));
}
