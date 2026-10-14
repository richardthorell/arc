#include <arc/editor/material_inspection.h>
#include <arc/editor/host_protocol.h>
#include <arc/editor/editor_state.h>
#include <arc/editor/entity_material_service.h>
#include <arc/editor/material_preview_realizer.h>
#include <arc/render_tools/material_graph.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <array>
#include <fstream>
#include <map>
#include <set>

namespace
{
using json = nlohmann::json;
constexpr auto material_guid = "11111111-1111-4111-8111-111111111111";
constexpr auto instance_guid = "22222222-2222-4222-8222-222222222222";

arc::editor::material_inspection_source material_source()
{
    return {.guid = material_guid, .path = "Content/custom.arcmat", .kind = "material", .source = R"({
      "version":4,"name":"Custom","domain":"surface","blendMode":"opaque","shadingModel":"standard",
      "graph":{"version":1,"nodes":[
        {"id":"output","type":"output","values":{}},
        {"id":"custom-value","type":"constant","values":{"value":0.4,"min":0,"max":1},
         "parameter":{"exposed":true,"name":"An entirely new control"}}
      ],"connections":[{"id":"link","from":{"nodeId":"custom-value","pin":"value"},
        "to":{"nodeId":"output","pin":"roughness"}}]}})"};
}

const arc::editor::material_inspection_resolver missing =
    [](auto, auto) -> std::optional<arc::editor::material_inspection_source> { return std::nullopt; };
} // namespace

TEST_CASE("material inspection reflects compiler IDs without display-name recipes")
{
    auto source = material_source();
    const auto result = json::parse(arc::editor::inspect_saved_material(source, missing));
    REQUIRE(result["definitionReady"] == true);
    REQUIRE(result["runtimeStatus"] == "notInspected");
    REQUIRE(result["provenance"] == "savedSource");
    REQUIRE(result["parameters"].size() == 1);
    const auto compiled = arc::render::tools::compile_material_graph_json(json::parse(source.source)["graph"].dump());
    REQUIRE(compiled);
    REQUIRE(result["parameters"][0]["id"] ==
            std::to_string(compiled.value().descriptor.parameters[0].id.representation()));
    REQUIRE(result["parameters"][0]["resolvedValue"][0].get<float>() == 0.4f);
    source.path = "Content/moved.arcmat";
    REQUIRE(json::parse(arc::editor::inspect_saved_material(source, missing))["definitionRevision"] ==
            result["definitionRevision"]);
    auto document = json::parse(source.source);
    document["graph"]["nodes"][1]["values"]["value"] = 0.6;
    source.source = document.dump();
    REQUIRE(json::parse(arc::editor::inspect_saved_material(source, missing))["definitionRevision"] !=
            result["definitionRevision"]);
}

TEST_CASE("material inspection resolves instance parents by GUID and includes their revisions")
{
    auto parent = material_source();
    const auto base = json::parse(arc::editor::inspect_saved_material(parent, missing));
    json document{{"version", 1},
                  {"name", "Instance"},
                  {"parent", {{"guid", material_guid}, {"pathHint", "old/location.arcmat"}}},
                  {"parameterOverrides", json::array({{{"parameterId", base["parameters"][0]["id"]}, {"value", 0.8}}})},
                  {"functionOverrides", json::array()}};
    arc::editor::material_inspection_source instance{
        .guid = instance_guid, .path = "instance.arcmatinst", .kind = "materialInstance", .source = document.dump()};
    const arc::editor::material_inspection_resolver resolve =
        [&](auto guid, auto) -> std::optional<arc::editor::material_inspection_source>
    { return guid == material_guid ? std::optional{parent} : std::nullopt; };
    const auto result = json::parse(arc::editor::inspect_saved_material(instance, resolve));
    REQUIRE(result["definitionReady"] == true);
    REQUIRE(result["parentGuid"] == material_guid);
    REQUIRE(result["parameters"][0]["resolvedValue"][0] == 0.8);
    parent.source += " ";
    REQUIRE(json::parse(arc::editor::inspect_saved_material(instance, resolve))["definitionRevision"] !=
            result["definitionRevision"]);
    document["parameterOverrides"][0]["value"] = "bad scalar";
    instance.source = document.dump();
    const auto invalid = json::parse(arc::editor::inspect_saved_material(instance, resolve));
    REQUIRE(invalid["definitionReady"] == false);
    REQUIRE_FALSE(invalid["diagnostics"].empty());
    REQUIRE(json::parse(arc::editor::inspect_saved_material(instance, missing))["definitionReady"] == false);
}

TEST_CASE("material inspection exposes authored presentation independently of output semantics")
{
    auto source = material_source();
    auto document = json::parse(source.source);
    auto& graph = document["graph"];
    graph["groups"] = json::array({{{"id", "surface"}, {"name", "Surface controls"}, {"order", 4}}});
    auto& node = graph["nodes"][1];
    node["type"] = "colorRgba";
    node["values"] = {{"value", json::array({0.1, 0.2, 0.3, 1.0})}};
    node["parameter"]["group"] = "surface";
    node["parameter"]["order"] = 7;
    graph["connections"][0]["to"]["pin"] = "emissive";
    source.source = document.dump();
    const auto result = json::parse(arc::editor::inspect_saved_material(source, missing));
    INFO(result.dump(2));
    REQUIRE(result["definitionReady"] == true);
    const auto& parameter = result["parameters"][0];
    CHECK(parameter["semantic"] == "none");
    CHECK(parameter["presentation"] == json({{"editorKind", "color"},
                                             {"groupId", "surface"},
                                             {"groupLabel", "Surface controls"},
                                             {"groupOrder", 4},
                                             {"order", 7}}));
    graph["nodes"][1]["type"] = "vector4";
    source.source = document.dump();
    const auto vector_result = json::parse(arc::editor::inspect_saved_material(source, missing));
    REQUIRE(vector_result["definitionReady"] == true);
    CHECK(vector_result["parameters"][0]["id"] == parameter["id"]);
    CHECK(vector_result["parameters"][0]["presentation"]["editorKind"] == "vector");
}

TEST_CASE("material inspection rejects malformed and oversized definitions without mutation")
{
    auto source = material_source();
    source.source = "not json";
    REQUIRE(json::parse(arc::editor::inspect_saved_material(source, missing))["definitionReady"] == false);
    source.source.assign(8 * 1024 * 1024 + 1, ' ');
    const auto oversized = json::parse(arc::editor::inspect_saved_material(source, missing));
    REQUIRE(oversized["definitionReady"] == false);
    REQUIRE_FALSE(oversized["diagnostics"].empty());
}

TEST_CASE("saved built-in instances expose native function selections and inherited input values")
{
    std::map<std::string, arc::editor::material_inspection_source> sources;
    const auto root = std::filesystem::path(ARC_SOURCE_ROOT) / "assets";
    for (const auto& file : std::filesystem::recursive_directory_iterator(root))
    {
        const auto extension = file.path().extension().string();
        if (extension != ".arcmat" && extension != ".arcmatinst" && extension != ".arcmatfn") continue;
        std::ifstream metadata(file.path().string() + ".arcmeta");
        if (!metadata) continue;
        json identity;
        metadata >> identity;
        std::ifstream stream(file.path());
        std::string source{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
        const auto path = file.path().lexically_relative(root).generic_string();
        sources.emplace(path, arc::editor::material_inspection_source{.guid = identity.at("guid").get<std::string>(),
                                                                      .path = path,
                                                                      .kind = extension == ".arcmat" ? "material"
                                                                              : extension == ".arcmatinst"
                                                                                  ? "materialInstance"
                                                                                  : "materialFunction",
                                                                      .source = std::move(source)});
    }
    const arc::editor::material_inspection_resolver resolve =
        [&](auto guid, auto hint) -> std::optional<arc::editor::material_inspection_source>
    {
        if (!guid.empty())
        {
            for (const auto& [path, source] : sources)
                if (source.guid == guid) return source;
            return std::nullopt;
        }
        const auto found = sources.find(std::string(hint));
        return found == sources.end() ? std::nullopt : std::optional{found->second};
    };
    const auto result =
        json::parse(arc::editor::inspect_saved_material(sources.at("materials/floor.arcmatinst"), resolve));
    INFO(result.dump(2));
    REQUIRE(result["definitionReady"] == true);
    REQUIRE_FALSE(result["functionSlots"].empty());
    REQUIRE_FALSE(result["dependencies"].empty());
    bool found_checker = false;
    for (const auto& slot : result["functionSlots"])
        if (slot["id"] == "base-color-source")
        {
            found_checker = true;
            REQUIRE(slot["selectedFunctionGuid"] == sources.at("material_functions/checker.arcmatfn").guid);
            CHECK(slot["inheritedFunctionGuid"] == slot["selectedFunctionGuid"]);
            CHECK(slot["overridden"] == false);
            REQUIRE(slot["alternatives"].size() >= 2);
        }
    REQUIRE(found_checker);
}

TEST_CASE("material and asset inspection queries round trip GUID identity")
{
    for (const auto* type : {"materials.describe", "assets.inspect"})
    {
        arc::editor::host_query_envelope query;
        std::string error;
        const auto wire =
            json{{"kind", "query"}, {"requestId", 42}, {"type", type}, {"payload", {{"guid", material_guid}}}}.dump();
        REQUIRE(arc::editor::from_json(wire, query, error));
        REQUIRE(arc::editor::query_type(query.payload) == type);
        REQUIRE(json::parse(arc::editor::to_json(query))["payload"]["guid"] == material_guid);
    }
}

TEST_CASE("function discovery query preserves bounded continuation identity")
{
    arc::editor::host_query_envelope query;
    std::string error;
    json wire{{"kind", "query"},
              {"requestId", 43},
              {"type", "materials.describe"},
              {"payload",
               {{"guid", material_guid},
                {"functionSlotId", "source"},
                {"functionOffset", 5},
                {"functionLimit", 5},
                {"discoveryRevision", std::string(64, 'a')}}}};
    REQUIRE(arc::editor::from_json(wire.dump(), query, error));
    const auto round_trip = json::parse(arc::editor::to_json(query))["payload"];
    CHECK(round_trip["functionSlotId"] == "source");
    CHECK(round_trip["functionOffset"] == 5);
    CHECK(round_trip["functionLimit"] == 5);
    CHECK(round_trip["discoveryRevision"] == std::string(64, 'a'));
    wire["payload"].erase("discoveryRevision");
    CHECK_FALSE(arc::editor::from_json(wire.dump(), query, error));
    wire["payload"]["functionOffset"] = 0;
    wire["payload"]["functionLimit"] = 17;
    CHECK_FALSE(arc::editor::from_json(wire.dump(), query, error));
}

TEST_CASE("function compatibility is independent of the bounded discovery preview")
{
    auto source = material_source();
    auto document = json::parse(source.source);
    std::map<std::string, arc::editor::material_inspection_source> functions;
    json choices = json::array();
    for (int index = 0; index < 17; ++index)
    {
        const auto path = "functions/f" + std::to_string(index) + ".arcmatfn";
        const auto guid = "00000000-0000-4000-8000-" + std::to_string(100000000000LL + index);
        functions.emplace(path, arc::editor::material_inspection_source{
                                    .guid = guid,
                                    .path = path,
                                    .kind = "materialFunction",
                                    .source = R"({"kind":"materialFunction","version":1,"name":"Choice",
              "inputs":[],"outputs":[{"id":"result","name":"Result","type":"float"}],
              "graph":{"version":1,"nodes":[{"id":"value","type":"constant","values":{"value":0.5}},
                {"id":"out","type":"functionOutput","values":{}}],
                "connections":[{"from":{"nodeId":"value","pin":"value"},"to":{"nodeId":"out","pin":"result"}}]}})"});
        choices.push_back({{"path", path}});
    }
    document["graph"]["nodes"][1] = {
        {"id", "custom-value"},
        {"type", "functionCall"},
        {"values",
         {{"slotId", "source"}, {"name", "Source"}, {"path", "functions/f0.arcmatfn"}, {"functions", choices}}}};
    document["graph"]["connections"][0]["from"]["pin"] = "result";
    source.source = document.dump();
    const arc::editor::material_inspection_resolver resolve =
        [&](auto guid, auto hint) -> std::optional<arc::editor::material_inspection_source>
    {
        for (const auto& [path, function] : functions)
            if (guid.empty() ? path == hint : function.guid == guid) return function;
        return std::nullopt;
    };
    const auto preview = json::parse(arc::editor::inspect_saved_material(source, resolve));
    REQUIRE(preview["definitionReady"] == true);
    const auto& slot = preview["functionSlots"][0];
    REQUIRE(slot["alternatives"].size() == 16);
    REQUIRE(slot["alternativesTruncated"] == true);
    const auto& last = functions.at("functions/f16.arcmatfn");
    for (const auto& alternative : slot["alternatives"])
        CHECK(alternative["guid"] != last.guid);
    const auto overrides = json::array(
        {{{"kind", "function"}, {"slotId", "source"}, {"function", {{"guid", last.guid}, {"pathHint", last.path}}}}});
    const auto selected = json::parse(arc::editor::inspect_saved_material(source, resolve, overrides.dump()));
    INFO(selected.dump(2));
    REQUIRE(selected["definitionReady"] == true);
    CHECK(selected["functionSlots"][0]["selectedFunctionGuid"] == last.guid);
    std::set<std::string> discovered;
    arc::editor::material_function_discovery_request request{.slot_id = "source", .limit = 5};
    for (int page = 0; page < 4; ++page)
    {
        const auto response = json::parse(arc::editor::inspect_saved_material(source, resolve, "[]", request));
        INFO(response.dump(2));
        REQUIRE(response["definitionReady"] == true);
        const auto& result = response["functionDiscovery"];
        REQUIRE(result["alternatives"].size() <= 5);
        for (const auto& alternative : result["alternatives"])
            CHECK(discovered.insert(alternative["guid"].get<std::string>()).second);
        request.revision = result["discoveryRevision"].get<std::string>();
        if (page == 3)
            CHECK(result["nextOffset"].is_null());
        else
        {
            REQUIRE_FALSE(result["nextOffset"].is_null());
            request.offset = result["nextOffset"].get<std::uint32_t>();
        }
    }
    CHECK(discovered.size() == 17);
    CHECK(discovered.contains(last.guid));
    const auto changed_selection =
        json::parse(arc::editor::inspect_saved_material(source, resolve, overrides.dump(), request));
    CHECK(changed_selection["definitionReady"] == false);
    source.source += " ";
    const auto stale = json::parse(arc::editor::inspect_saved_material(source, resolve, "[]", request));
    CHECK(stale["definitionReady"] == false);
    CHECK(stale["diagnostics"][0]["message"].get<std::string>().find("Stale") != std::string::npos);
    request = {.slot_id = "missing"};
    CHECK(json::parse(arc::editor::inspect_saved_material(source, resolve, "[]", request))["definitionReady"] == false);
    request = {.slot_id = "source", .limit = 0};
    CHECK(json::parse(arc::editor::inspect_saved_material(source, resolve, "[]", request))["definitionReady"] == false);
}

TEST_CASE("entity material inspection separates inherited and local values without changing definition identity")
{
    const auto source = material_source();
    const auto saved = json::parse(arc::editor::inspect_saved_material(source, missing));
    const auto id = saved["parameters"][0]["id"];
    auto overrides = json::array({{{"kind", "scalar"}, {"parameterId", id}, {"value", json::array({0.7})}}});
    const auto local = json::parse(arc::editor::inspect_saved_material(source, missing, overrides.dump()));
    REQUIRE(local["definitionReady"] == true);
    REQUIRE(local["definitionRevision"] == saved["definitionRevision"]);
    REQUIRE(local["parameters"][0]["overridden"] == true);
    CHECK(local["parameters"][0]["localValue"] == json::array({0.7}));
    CHECK(saved["parameters"][0]["localValue"].is_null());
    REQUIRE(local["parameters"][0]["resolvedValue"][0] == 0.7);
    REQUIRE(local["parameters"][0]["inheritedValue"] == saved["parameters"][0]["resolvedValue"]);
    overrides[0]["value"] = json::array({2.0});
    const auto invalid = json::parse(arc::editor::inspect_saved_material(source, missing, overrides.dump()));
    REQUIRE(invalid["definitionReady"] == true);
    REQUIRE(invalid["parameters"][0]["overridden"] == false);
    CHECK(invalid["parameters"][0]["localValue"].is_null());
    REQUIRE(invalid["diagnostics"][0]["code"] == "stale-entity-override");
    overrides[0]["parameterId"] = "999999";
    const auto stale = json::parse(arc::editor::inspect_saved_material(source, missing, overrides.dump()));
    REQUIRE(stale["diagnostics"][0]["code"] == "stale-entity-override");
}

TEST_CASE("typed material edit commands round trip revision authority and lossless parameter identity")
{
    const auto wire = json{{"kind", "command"},
                           {"requestId", 51},
                           {"type", "materials.edit"},
                           {"payload",
                            {{"operation", "setOverride"},
                             {"entityGuid", material_guid},
                             {"definitionRevision", std::string(64, 'a')},
                             {"expectedSceneRevision", 7},
                             {"worldEpoch", 3},
                             {"parameterId", "18446744073709551615"},
                             {"value", json::array({0.25})}}}};
    arc::editor::host_command_envelope command;
    std::string error;
    REQUIRE(arc::editor::from_json(wire.dump(), command, error));
    const auto roundtrip = json::parse(arc::editor::to_json(command));
    REQUIRE(roundtrip["payload"]["parameterId"] == "18446744073709551615");
    REQUIRE(roundtrip["payload"]["expectedSceneRevision"] == 7);
    REQUIRE(roundtrip["payload"]["worldEpoch"] == 3);
}

TEST_CASE("typed entity overrides isolate shared materials and retain last good values on rejection")
{
    arc::render::renderer renderer;
    arc::editor::editor_scene_state scene;
    const auto source = material_source();
    const auto realized = arc::editor::realize_material_preview_descriptor(source.source, "Custom");
    REQUIRE(realized.succeeded);
    const auto shared = renderer.create_material(realized.material);
    auto asset = arc::editor::make_default_material_asset("Custom");
    asset.material = realized.material;
    asset.path = source.path;
    scene.material_library.materials.push_back({source.path, asset, shared});
    const auto first = scene.scene.create();
    const auto second = scene.scene.create();
    scene.scene.emplace<arc::scene::mesh_renderer_component>(first,
                                                             arc::scene::mesh_renderer_component{.material = shared});
    scene.scene.emplace<arc::scene::mesh_renderer_component>(second,
                                                             arc::scene::mesh_renderer_component{.material = shared});
    arc::editor::ensure_scene_authoring_metadata(scene);
    const auto parameter = realized.material.runtime_program->parameters.front().id;
    const std::array values{0.75f};
    std::string error;
    REQUIRE(
        arc::editor::set_entity_material_parameter(scene, renderer, first, parameter, values, nullptr, false, error));
    const auto local = scene.scene.try_get<arc::scene::mesh_renderer_component>(first)->material;
    REQUIRE(local != shared);
    REQUIRE(scene.scene.try_get<arc::scene::mesh_renderer_component>(second)->material == shared);
    const auto saved = arc::editor::entity_material_overrides_json(scene, first);
    REQUIRE_FALSE(arc::editor::set_entity_material_parameter(scene, renderer, first, parameter, {}, nullptr, false,
                                                             error, {}, true));
    REQUIRE(arc::editor::entity_material_overrides_json(scene, first) == saved);
    REQUIRE(json::parse(saved)[0]["value"][0] == 0.75f);
    REQUIRE(arc::editor::entity_material_overrides_json(scene, second) == "[]");
    const std::array invalid{2.0f};
    REQUIRE_FALSE(
        arc::editor::set_entity_material_parameter(scene, renderer, first, parameter, invalid, nullptr, false, error));
    REQUIRE_FALSE(error.empty());
    REQUIRE(scene.scene.try_get<arc::scene::mesh_renderer_component>(first)->material == local);
    REQUIRE(arc::editor::entity_material_overrides_json(scene, first) == saved);
    REQUIRE(arc::editor::set_entity_material_parameter(scene, renderer, first, parameter, {}, nullptr, true, error));
    REQUIRE(scene.scene.try_get<arc::scene::mesh_renderer_component>(first)->material == shared);
    REQUIRE(arc::editor::entity_material_overrides_json(scene, first) == "[]");

    SECTION("function reset removes local inputs and selection without changing another entity")
    {
        REQUIRE(arc::editor::set_entity_material_parameter(scene, renderer, first, parameter, values, nullptr, false,
                                                           error, "surface-source"));
        const auto entity_guid = arc::editor::entity_guid_of(scene, first);
        for (auto& record : scene.preserved_component_records)
        {
            if (record.entity != entity_guid || record.component_name != "MeshRenderer") continue;
            auto data = json::parse(record.json);
            data["materialParameterOverrides"].push_back(
                {{"kind", "function"},
                 {"slotId", "surface-source"},
                 {"function", {{"guid", instance_guid}, {"pathHint", "old-function.arcmatfn"}}}});
            record.json = data.dump();
        }
        REQUIRE(json::parse(arc::editor::entity_material_overrides_json(scene, first)).size() == 2);
        REQUIRE(arc::editor::reset_entity_material_function(scene, renderer, first, "surface-source", error));
        CHECK(arc::editor::entity_material_overrides_json(scene, first) == "[]");
        CHECK(scene.scene.try_get<arc::scene::mesh_renderer_component>(first)->material == shared);
        CHECK(scene.scene.try_get<arc::scene::mesh_renderer_component>(second)->material == shared);
        CHECK_FALSE(arc::editor::reset_entity_material_function(scene, renderer, first, "", error));
        CHECK(scene.scene.try_get<arc::scene::mesh_renderer_component>(first)->material == shared);
    }
}

TEST_CASE("typed clear and function reset commands round trip without replacement values")
{
    for (const auto* operation : {"clearTexture", "resetFunction"})
    {
        auto payload = json{{"operation", operation},
                            {"entityGuid", material_guid},
                            {"definitionRevision", std::string(64, 'a')},
                            {"expectedSceneRevision", 7},
                            {"worldEpoch", 3}};
        payload[std::string_view(operation) == "clearTexture" ? "parameterId" : "slotId"] = "123";
        auto wire = json{{"kind", "command"}, {"requestId", 51}, {"type", "materials.edit"}, {"payload", payload}};
        arc::editor::host_command_envelope command;
        std::string error;
        REQUIRE(arc::editor::from_json(wire.dump(), command, error));
        const auto serialized = arc::editor::to_json(command);
        CHECK(json::parse(serialized)["payload"] == payload);
        REQUIRE(arc::editor::from_json(serialized, command, error));
        wire["payload"]["value"] = nullptr;
        CHECK_FALSE(arc::editor::from_json(wire.dump(), command, error));
    }
}

TEST_CASE("texture inspection distinguishes GUID defaults inheritance empty and unresolved values")
{
    auto source = material_source();
    auto document = json::parse(source.source);
    auto& graph = document["graph"];
    graph["nodes"][1]["type"] = "textureSample";
    graph["nodes"][1]["values"] = {{"texture", "textures/albedo.png"}};
    graph["connections"][0]["from"]["pin"] = "rgb";
    graph["connections"][0]["to"]["pin"] = "baseColor";
    source.source = document.dump();
    arc::editor::material_inspection_source texture{
        .guid = instance_guid, .path = "textures/albedo.png", .kind = "texture2d"};
    const arc::editor::material_inspection_resolver resolve =
        [&](auto guid, auto hint) -> std::optional<arc::editor::material_inspection_source>
    {
        if (guid == material_guid) return source;
        if (guid.empty() ? hint == texture.path : guid == texture.guid) return texture;
        return std::nullopt;
    };
    const auto saved = json::parse(arc::editor::inspect_saved_material(source, resolve));
    INFO(saved.dump(2));
    REQUIRE(saved["definitionReady"] == true);
    const auto& parameter = saved["parameters"][0];
    CHECK(parameter["defaultValue"]["guid"] == texture.guid);
    CHECK(parameter["texture"]["defaultStatus"] == "resolved");
    CHECK(parameter["texture"]["dimension"] == "2d");
    const auto cleared = json::array({{{"kind", "texture"}, {"parameterId", parameter["id"]}, {"texture", ""}}});
    const auto empty =
        json::parse(arc::editor::inspect_saved_material(source, resolve, cleared.dump()))["parameters"][0];
    CHECK(empty["texture"]["defaultStatus"] == "resolved");
    CHECK(empty["texture"]["inheritedStatus"] == "resolved");
    CHECK(empty["texture"]["effectiveStatus"] == "empty");
    CHECK(empty["overridden"] == true);
    CHECK(empty["valueStatus"] == "resolved");
    auto local = cleared;
    local[0]["textureReference"] = {{"guid", texture.guid}, {"pathHint", "old/name.png"}};
    const auto bound = json::parse(arc::editor::inspect_saved_material(source, resolve, local.dump()));
    CHECK(bound["parameters"][0]["localValue"]["guid"] == texture.guid);
    CHECK(bound["parameters"][0]["texture"]["effectiveStatus"] == "resolved");
    local[0]["textureReference"]["guid"] = "44444444-4444-4444-8444-444444444444";
    local[0]["textureReference"]["pathHint"] = texture.path;
    const auto stale = json::parse(arc::editor::inspect_saved_material(source, resolve, local.dump()));
    CHECK(stale["parameters"][0]["texture"]["effectiveStatus"] == "unresolved");
    CHECK(stale["parameters"][0]["overridden"] == true);
    CHECK(stale["parameters"][0]["texture"]["inheritedStatus"] == "resolved");
    const auto unresolved = json::parse(arc::editor::inspect_saved_material(source, missing));
    CHECK(unresolved["definitionReady"] == true);
    CHECK(unresolved["parameters"][0]["texture"]["defaultStatus"] == "unresolved");
    CHECK(unresolved["parameters"][0]["valueStatus"] == "unavailable");
    REQUIRE_FALSE(unresolved["diagnostics"].empty());

    json instance_document{{"version", 1},
                           {"name", "Instance"},
                           {"parent", {{"guid", material_guid}}},
                           {"parameterOverrides", json::array({{{"parameterId", parameter["id"]}, {"value", ""}}})},
                           {"functionOverrides", json::array()}};
    arc::editor::material_inspection_source instance{.guid = "33333333-3333-4333-8333-333333333333",
                                                     .path = "instance.arcmatinst",
                                                     .kind = "materialInstance",
                                                     .source = instance_document.dump()};
    const auto inherited_empty = json::parse(arc::editor::inspect_saved_material(instance, resolve));
    INFO(inherited_empty.dump(2));
    REQUIRE(inherited_empty["definitionReady"] == true);
    CHECK(inherited_empty["parameters"][0]["texture"]["inheritedStatus"] == "empty");
    CHECK(inherited_empty["parameters"][0]["overridden"] == false);

    graph["nodes"][1]["values"]["texture"] = texture.guid;
    source.source = document.dump();
    texture.path = "textures/renamed.png";
    const auto renamed = json::parse(arc::editor::inspect_saved_material(source, resolve));
    CHECK(renamed["parameters"][0]["defaultValue"]["pathHint"] == texture.path);
    texture.kind = "material";
    const auto incompatible = json::parse(arc::editor::inspect_saved_material(source, resolve));
    CHECK(incompatible["parameters"][0]["texture"]["defaultStatus"] == "unresolved");
}

TEST_CASE("expanded function textures retain compiler default sources")
{
    auto source = material_source();
    auto document = json::parse(source.source);
    document["graph"]["nodes"][1] = {{"id", "call"},
                                     {"type", "functionCall"},
                                     {"values", {{"path", "functions/texture.arcmatfn"}, {"slotId", "surface"}}}};
    document["graph"]["connections"][0]["from"] = {{"nodeId", "call"}, {"pin", "result"}};
    document["graph"]["connections"][0]["to"]["pin"] = "baseColor";
    source.source = document.dump();
    const arc::editor::material_inspection_source function{
        .guid = instance_guid, .path = "functions/texture.arcmatfn", .kind = "materialFunction", .source = R"({
        "kind":"materialFunction","version":1,"name":"Textured","inputs":[],
        "outputs":[{"id":"result","name":"Result","type":"vec3"}],
        "graph":{"version":1,"nodes":[
          {"id":"image","type":"textureSample","values":{"texture":"images/color.png"},
           "parameter":{"exposed":true,"name":"Image"}},
          {"id":"out","type":"functionOutput","values":{}}],
        "connections":[{"from":{"nodeId":"image","pin":"rgb"},"to":{"nodeId":"out","pin":"result"}}]}})"};
    const arc::editor::material_inspection_source texture{
        .guid = "33333333-3333-4333-8333-333333333333", .path = "images/color.png", .kind = "texture2d"};
    const arc::editor::material_inspection_resolver resolve =
        [&](auto guid, auto hint) -> std::optional<arc::editor::material_inspection_source>
    {
        if (guid.empty() ? hint == function.path : guid == function.guid) return function;
        if (guid.empty() ? hint == texture.path : guid == texture.guid) return texture;
        return std::nullopt;
    };
    const auto inspected = json::parse(arc::editor::inspect_saved_material(source, resolve));
    INFO(inspected.dump(2));
    REQUIRE(inspected["definitionReady"] == true);
    REQUIRE(inspected["parameters"].size() == 1);
    CHECK(inspected["parameters"][0]["defaultValue"]["guid"] == texture.guid);
    const std::vector<arc::render::tools::material_function_source> functions{
        {.path = function.path, .identity = function.guid, .source = function.source}};
    const auto compiled = arc::render::tools::compile_material_graph_json(document["graph"].dump(), functions);
    REQUIRE(compiled);
    REQUIRE(compiled.value().descriptor.textures.size() == 1);
    CHECK(compiled.value().descriptor.textures[0].default_source == texture.path);
}

TEST_CASE("typed texture clearing persists an empty override while reset restores inheritance")
{
    auto source = material_source();
    auto document = json::parse(source.source);
    auto& graph = document["graph"];
    graph["nodes"][1]["type"] = "textureSample";
    graph["nodes"][1]["values"] = json::object();
    graph["connections"][0]["from"]["pin"] = "rgb";
    graph["connections"][0]["to"]["pin"] = "baseColor";
    source.source = document.dump();
    arc::render::renderer renderer;
    arc::editor::editor_scene_state scene;
    const auto realized = arc::editor::realize_material_preview_descriptor(source.source, "Texture material");
    INFO(realized.message);
    REQUIRE(realized.succeeded);
    const auto shared = renderer.create_material(realized.material);
    auto asset = arc::editor::make_default_material_asset("Texture material");
    asset.material = realized.material;
    asset.path = source.path;
    scene.material_library.materials.push_back({source.path, asset, shared});
    const auto entity = scene.scene.create();
    scene.scene.emplace<arc::scene::mesh_renderer_component>(entity,
                                                             arc::scene::mesh_renderer_component{.material = shared});
    arc::editor::ensure_scene_authoring_metadata(scene);
    const auto parameter = arc::render::make_shader_parameter_id("custom-value");
    std::string error;
    REQUIRE(arc::editor::set_entity_material_parameter(scene, renderer, entity, parameter, {}, nullptr, false, error,
                                                       {}, true));
    const auto overrides = arc::editor::entity_material_overrides_json(scene, entity);
    REQUIRE(json::parse(overrides).size() == 1);
    CHECK(json::parse(overrides)[0]["texture"] == "");
    const auto inspected = json::parse(arc::editor::inspect_saved_material(source, missing, overrides));
    REQUIRE(inspected["definitionReady"] == true);
    CHECK(inspected["parameters"][0]["overridden"] == true);
    CHECK(inspected["parameters"][0]["resolvedValue"].is_null());
    CHECK(inspected["parameters"][0]["valueStatus"] == "resolved");
    const auto cleared_handle = scene.scene.try_get<arc::scene::mesh_renderer_component>(entity)->material;
    const arc::assets::asset_reference missing_texture{.guid = *arc::assets::parse_asset_guid(instance_guid),
                                                       .expected_type = arc::assets::asset_types::texture_2d};
    CHECK_FALSE(arc::editor::set_entity_material_parameter(scene, renderer, entity, parameter, {}, &missing_texture,
                                                           false, error));
    CHECK(arc::editor::entity_material_overrides_json(scene, entity) == overrides);
    CHECK(scene.scene.try_get<arc::scene::mesh_renderer_component>(entity)->material == cleared_handle);
    CHECK_FALSE(arc::editor::set_entity_material_parameter(scene, renderer, entity, parameter, {}, nullptr, true, error,
                                                           {}, true));
    CHECK(arc::editor::entity_material_overrides_json(scene, entity) == overrides);
    REQUIRE(arc::editor::set_entity_material_parameter(scene, renderer, entity, parameter, {}, nullptr, true, error));
    CHECK(arc::editor::entity_material_overrides_json(scene, entity) == "[]");
    CHECK(scene.scene.try_get<arc::scene::mesh_renderer_component>(entity)->material == shared);
}
