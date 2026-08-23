#include <arc/render_tools/material_asset.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

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

void append_parameter(std::vector<std::byte>& output, const shader_parameter_descriptor& parameter)
{
    append_value(output, parameter.id.representation());
    append_string(output, parameter.name);
    append_value(output, parameter.type);
    append_value(output, parameter.offset);
    append_value(output, parameter.size);
}

class package_reader
{
public:
    explicit package_reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <class T> bool value(T& output)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (cursor_ > bytes_.size() || sizeof(T) > bytes_.size() - cursor_) return false;
        std::memcpy(&output, bytes_.data() + cursor_, sizeof(T));
        cursor_ += sizeof(T);
        return true;
    }

    bool string(std::string& output)
    {
        std::uint64_t size{};
        if (!value(size) || size > static_cast<std::uint64_t>(bytes_.size() - cursor_)) return false;
        output.assign(reinterpret_cast<const char*>(bytes_.data() + cursor_), static_cast<std::size_t>(size));
        cursor_ += static_cast<std::size_t>(size);
        return true;
    }

    bool raw(std::span<std::byte> output)
    {
        if (cursor_ > bytes_.size() || output.size() > bytes_.size() - cursor_) return false;
        std::memcpy(output.data(), bytes_.data() + cursor_, output.size());
        cursor_ += output.size();
        return true;
    }

    [[nodiscard]] bool complete() const noexcept
    {
        return cursor_ == bytes_.size();
    }

private:
    std::span<const std::byte> bytes_;
    std::size_t cursor_{};
};

bool read_parameter(package_reader& reader, shader_parameter_descriptor& parameter)
{
    std::uint64_t id{};
    if (!reader.value(id) || !reader.string(parameter.name) || !reader.value(parameter.type) ||
        !reader.value(parameter.offset) || !reader.value(parameter.size))
        return false;
    parameter.id = {id};
    return parameter.id.valid();
}

material_domain authored_domain(const json& document)
{
    const auto domain = document.value("domain", std::string{"surface"});
    return domain == "terrain" ? material_domain::terrain : material_domain::surface;
}

material_shading_model authored_shading_model(const json& document)
{
    const auto model = document.value("shadingModel", std::string{"standard"});
    if (model == "skin") return material_shading_model::skin;
    if (model == "transmission") return material_shading_model::transmission;
    if (model == "unlit") return material_shading_model::unlit;
    if (model == "customLit" || model == "custom_lit") return material_shading_model::custom_lit;
    return material_shading_model::standard;
}

material_alpha_mode authored_alpha_mode(const json& document)
{
    const auto mode = document.value("blendMode", std::string{"opaque"});
    if (mode == "masked") return material_alpha_mode::masked;
    if (mode == "blend") return material_alpha_mode::blend;
    return material_alpha_mode::opaque;
}

} // namespace

material_authoring_result parse_material_authoring_json(std::string_view source)
{
    auto document = json::parse(source, nullptr, false);
    if (document.is_discarded() || !document.is_object())
        return material_authoring_result::failure(
            {.code = material_asset_error_code::malformed_json, .message = "Material definition is not valid JSON"});

    if (!document.contains("version") || !document["version"].is_number_integer())
        return material_authoring_result::failure({.code = material_asset_error_code::invalid_document,
                                                   .message = "Material document version must be an integer"});
    const auto authored_version = document["version"].get<std::int64_t>();
    if (authored_version != static_cast<std::int64_t>(material_authoring_version))
        return material_authoring_result::failure(
            {.code = material_asset_error_code::unsupported_version,
             .message = "Material document must use schema v" + std::to_string(material_authoring_version) +
                        "; legacy material schemas are no longer supported"});

    std::string graph_json;
    if (document.contains("graph") && !document["graph"].is_null())
    {
        if (!document["graph"].is_object())
            return material_authoring_result::failure({.code = material_asset_error_code::invalid_document,
                                                       .message = "Material document graph must be an object or null"});
        graph_json = document["graph"].dump();
    }

    std::string shader_path;
    if (document.contains("shaderPath") && !document["shaderPath"].is_null())
    {
        if (!document["shaderPath"].is_string())
            return material_authoring_result::failure(
                {.code = material_asset_error_code::invalid_document,
                 .message = "Material document shaderPath must be a string or null"});
        shader_path = document["shaderPath"].get<std::string>();
        if (shader_path.empty())
            return material_authoring_result::failure({.code = material_asset_error_code::invalid_document,
                                                       .message = "Material Shader path must not be empty"});
    }

    const bool has_graph = !graph_json.empty();
    const bool has_shader = !shader_path.empty();
    if (has_graph == has_shader)
        return material_authoring_result::failure(
            {.code = material_asset_error_code::invalid_document,
             .message = has_graph ? "Material must use either a graph or shaderPath, not both"
                                  : "Material must provide a compiled graph or shaderPath"});

    return material_authoring_result::success({.source_version = material_authoring_version,
                                               .version = material_authoring_version,
                                               .migrated = false,
                                               .canonical_json = document.dump(),
                                               .graph_json = std::move(graph_json),
                                               .shader_path = std::move(shader_path),
                                               .domain = authored_domain(document),
                                               .shading_model = authored_shading_model(document),
                                               .alpha_mode = authored_alpha_mode(document),
                                               .double_sided = document.value("doubleSided", false)});
}

std::vector<std::byte> serialize_material_package_v3(const material_package_v3& package)
{
    std::vector<std::byte> output;
    append_string(output, material_package_signature);
    append_value(output, package.compiled.contract_version);
    append_value(output, package.compiled.material_abi);
    append_value(output, package.compiled.package.high);
    append_value(output, package.compiled.package.low);

    auto passes = package.compiled.passes;
    std::ranges::sort(passes, {}, &material_pass_binding::pass);
    append_value(output, static_cast<std::uint32_t>(passes.size()));
    for (const auto& pass : passes)
    {
        append_value(output, pass.pass);
        append_value(output, pass.permutation.representation());
        append_value(output, pass.entry_point.representation());
        output.insert(output.end(), pass.build_hash.bytes.begin(), pass.build_hash.bytes.end());
    }

    append_value(output, static_cast<std::uint32_t>(package.parameters.size()));
    for (const auto& parameter : package.parameters)
        append_parameter(output, parameter);
    append_string(output, package.canonical_document_json);
    return output;
}

material_package_v3_result deserialize_material_package_v3(std::span<const std::byte> bytes)
{
    package_reader reader(bytes);
    std::string signature;
    material_package_v3 package;
    if (!reader.string(signature) || signature != material_package_signature ||
        !reader.value(package.compiled.contract_version) || !reader.value(package.compiled.material_abi) ||
        !reader.value(package.compiled.package.high) || !reader.value(package.compiled.package.low))
        return material_package_v3_result::failure(
            {.code = material_asset_error_code::corrupt_package, .message = "Material package header is invalid"});

    if (package.compiled.contract_version != material_pass_contract_version ||
        package.compiled.material_abi != material_abi_version)
        return material_package_v3_result::failure(
            {.code = material_asset_error_code::unsupported_version,
             .message = "Material package uses an unsupported pass contract or Material ABI"});

    std::uint32_t pass_count{};
    if (!reader.value(pass_count) || pass_count == 0 || pass_count > 32u)
        return material_package_v3_result::failure(
            {.code = material_asset_error_code::corrupt_package,
             .message = "Material package must contain compiled pass bindings"});
    if (!package.compiled.package.valid())
        return material_package_v3_result::failure(
            {.code = material_asset_error_code::corrupt_package,
             .message = "Compiled material passes require a valid shader package ID"});

    package.compiled.passes.reserve(pass_count);
    for (std::uint32_t index = 0; index < pass_count; ++index)
    {
        material_pass_binding binding;
        std::uint64_t permutation{};
        std::uint64_t entry_point{};
        if (!reader.value(binding.pass) || !reader.value(permutation) || !reader.value(entry_point) ||
            !reader.raw(binding.build_hash.bytes))
            return material_package_v3_result::failure({.code = material_asset_error_code::corrupt_package,
                                                        .message = "Material package pass entry is invalid"});
        binding.permutation = {permutation};
        binding.entry_point = {entry_point};
        if (!binding.permutation.valid() || !binding.entry_point.valid() ||
            find_material_pass_binding(package.compiled, binding.pass) != nullptr)
            return material_package_v3_result::failure(
                {.code = material_asset_error_code::corrupt_package,
                 .message = "Material package contains an invalid or duplicate pass binding"});
        package.compiled.passes.push_back(binding);
    }

    std::uint32_t parameter_count{};
    if (!reader.value(parameter_count) || parameter_count > 65'536u)
        return material_package_v3_result::failure(
            {.code = material_asset_error_code::corrupt_package, .message = "Material parameter table is invalid"});
    package.parameters.reserve(parameter_count);
    for (std::uint32_t index = 0; index < parameter_count; ++index)
    {
        shader_parameter_descriptor parameter;
        if (!read_parameter(reader, parameter))
            return material_package_v3_result::failure(
                {.code = material_asset_error_code::corrupt_package, .message = "Material parameter entry is invalid"});
        package.parameters.push_back(std::move(parameter));
    }

    if (!reader.string(package.canonical_document_json) || !reader.complete())
        return material_package_v3_result::failure(
            {.code = material_asset_error_code::corrupt_package, .message = "Material package payload is truncated"});
    return material_package_v3_result::success(std::move(package));
}

} // namespace arc::render::tools
