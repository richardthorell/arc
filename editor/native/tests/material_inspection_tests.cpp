#include <arc/editor/material_inspection.h>
#include <arc/editor/host_protocol.h>
#include <arc/render_tools/material_graph.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <map>

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
