#include <arc/render_tools/material_asset.h>

#include <nlohmann/json.hpp>

#include <type_traits>

namespace arc::render::tools
{
namespace
{
using json = nlohmann::json;

template <class T> void append_value(std::vector<std::byte>& output, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* bytes = reinterpret_cast<const std::byte*>(&value);
    output.insert(output.end(), bytes, bytes + sizeof(T));
}

void append_string(std::vector<std::byte>& output, std::string_view value)
{
    append_value(output, static_cast<std::uint64_t>(value.size()));
    output.insert(output.end(), reinterpret_cast<const std::byte*>(value.data()),
                  reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

} // namespace

material_authoring_result parse_material_authoring_json(std::string_view source)
{
    auto document = json::parse(source, nullptr, false);
    if (document.is_discarded() || !document.is_object())
        return material_authoring_result::failure(
            {.code = material_asset_error_code::malformed_json, .message = "Material definition is not valid JSON"});

    const bool has_version = document.contains("version");
    std::uint32_t source_version = 1;
    if (has_version)
    {
        if (!document["version"].is_number_integer())
            return material_authoring_result::failure({.code = material_asset_error_code::invalid_document,
                                                       .message = "Material document version must be an integer"});
        const auto authored_version = document["version"].get<std::int64_t>();
        if (authored_version < 1 || authored_version > static_cast<std::int64_t>(material_authoring_version))
            return material_authoring_result::failure(
                {.code = material_asset_error_code::unsupported_version,
                 .message = "Unsupported material document version: " + std::to_string(authored_version)});
        source_version = static_cast<std::uint32_t>(authored_version);
    }

    std::string graph_json;
    if (document.contains("graph"))
    {
        if (!document["graph"].is_object())
            return material_authoring_result::failure({.code = material_asset_error_code::invalid_document,
                                                       .message = "Material document graph must be an object"});
        graph_json = document["graph"].dump();
    }

    std::string shader_path;
    if (document.contains("shaderPath"))
    {
        if (!document["shaderPath"].is_string())
            return material_authoring_result::failure({.code = material_asset_error_code::invalid_document,
                                                       .message = "Material document shaderPath must be a string"});
        shader_path = document["shaderPath"].get<std::string>();
    }

    document["version"] = material_authoring_version;
    return material_authoring_result::success({.source_version = source_version,
                                               .version = material_authoring_version,
                                               .migrated = !has_version || source_version != material_authoring_version,
                                               .canonical_json = document.dump(),
                                               .graph_json = std::move(graph_json),
                                               .shader_path = std::move(shader_path)});
}

std::vector<std::byte> serialize_material_package_v2(const material_package_v2& package)
{
    std::vector<std::byte> output;
    append_string(output, material_package_signature);
    append_value(output, package.shader_package.high);
    append_value(output, package.shader_package.low);
    append_value(output, package.permutation.representation());
    append_value(output, static_cast<std::uint32_t>(package.parameters.size()));
    for (const auto& parameter : package.parameters)
    {
        append_value(output, parameter.id.representation());
        append_string(output, parameter.name);
        append_value(output, parameter.type);
        append_value(output, parameter.offset);
        append_value(output, parameter.size);
    }
    append_string(output, package.canonical_document_json);
    return output;
}

} // namespace arc::render::tools
