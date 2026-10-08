#include <arc/render_tools/material_asset.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>

namespace arc::render::tools
{
namespace
{
using json = nlohmann::json;

constexpr std::array<std::string_view, 4> legacy_material_fields{"shader", "surface", "textures", "advanced"};

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
    append_value(output, parameter.has_range);
    append_value(output, parameter.minimum);
    append_value(output, parameter.maximum);
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
        !reader.value(parameter.offset) || !reader.value(parameter.size) || !reader.value(parameter.has_range) ||
        !reader.value(parameter.minimum) || !reader.value(parameter.maximum))
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

std::optional<shader_parameter_id> parse_parameter_id(std::string_view text)
{
    if (text.empty()) return std::nullopt;
    std::uint64_t value{};
    int base = 10;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X'))
    {
        text.remove_prefix(2);
        base = 16;
    }
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0) return std::nullopt;
    return shader_parameter_id{value};
}

bool parse_asset_reference(const json& value, material_instance_asset_reference& output)
{
    if (!value.is_object()) return false;
    output.guid = value.value("guid", "");
    output.path_hint = value.value("pathHint", "");
    return !output.guid.empty() && !output.path_hint.empty();
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
    {
        auto message = "Material document must use schema v" + std::to_string(material_authoring_version);
        message += "; legacy material schemas are no longer supported";
        return material_authoring_result::failure(
            {.code = material_asset_error_code::unsupported_version, .message = std::move(message)});
    }

    for (const auto field : legacy_material_fields)
        if (document.contains(field))
            return material_authoring_result::failure(
                {.code = material_asset_error_code::invalid_document,
                 .message = "Legacy material field '" + std::string(field) + "' is no longer supported"});

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

    return material_authoring_result::success({.version = material_authoring_version,
                                               .canonical_json = document.dump(),
                                               .graph_json = std::move(graph_json),
                                               .shader_path = std::move(shader_path),
                                               .domain = authored_domain(document),
                                               .shading_model = authored_shading_model(document),
                                               .alpha_mode = authored_alpha_mode(document),
                                               .double_sided = document.value("doubleSided", false),
                                               .cast_shadows = document.value("castShadows", true)});
}

material_instance_authoring_result parse_material_instance_authoring_json(std::string_view source)
{
    auto document = json::parse(source, nullptr, false);
    if (document.is_discarded() || !document.is_object())
        return material_instance_authoring_result::failure(
            {.code = material_asset_error_code::malformed_json,
             .message = "Material Instance definition is not valid JSON"});

    if (document.value("version", 0) != static_cast<int>(material_instance_authoring_version))
        return material_instance_authoring_result::failure(
            {.code = material_asset_error_code::unsupported_version,
             .message = "Material Instance document must use schema v" +
                        std::to_string(material_instance_authoring_version)});

    material_instance_authoring_document result;
    result.name = document.value("name", "");
    if (result.name.empty())
        return material_instance_authoring_result::failure(
            {.code = material_asset_error_code::invalid_document,
             .message = "Material Instance requires a non-empty name"});

    if (!document.contains("parent") || !parse_asset_reference(document["parent"], result.parent))
        return material_instance_authoring_result::failure(
            {.code = material_asset_error_code::invalid_document,
             .message = "Material Instance requires a GUID-backed parent Material reference"});

    const auto& parameter_overrides =
        document.contains("parameterOverrides") ? document["parameterOverrides"] : json::array();
    if (!parameter_overrides.is_array())
        return material_instance_authoring_result::failure(
            {.code = material_asset_error_code::invalid_document,
             .message = "Material Instance parameterOverrides must be an array"});

    std::set<std::uint64_t> parameter_ids;
    for (const auto& authored : parameter_overrides)
    {
        if (!authored.is_object() || !authored.contains("parameterId") || !authored["parameterId"].is_string() ||
            !authored.contains("value"))
            return material_instance_authoring_result::failure(
                {.code = material_asset_error_code::invalid_document,
                 .message = "Material Instance parameter override is malformed"});
        const auto parameter_id = parse_parameter_id(authored["parameterId"].get<std::string>());
        if (!parameter_id || !parameter_ids.insert(parameter_id->value).second)
            return material_instance_authoring_result::failure(
                {.code = material_asset_error_code::invalid_document,
                 .message = "Material Instance contains an invalid or duplicate parameter override"});
        result.parameter_overrides.push_back(
            {.parameter_id = *parameter_id, .value_json = authored["value"].dump()});
    }

    const auto& function_overrides =
        document.contains("functionOverrides") ? document["functionOverrides"] : json::array();
    if (!function_overrides.is_array())
        return material_instance_authoring_result::failure(
            {.code = material_asset_error_code::invalid_document,
             .message = "Material Instance functionOverrides must be an array"});

    std::set<std::string> slot_ids;
    for (const auto& authored : function_overrides)
    {
        if (!authored.is_object()) 
            return material_instance_authoring_result::failure(
                {.code = material_asset_error_code::invalid_document,
                 .message = "Material Instance Function Slot override is malformed"});
        material_instance_function_override_document override_value;
        override_value.slot_id = authored.value("slotId", "");
        if (override_value.slot_id.empty() || !slot_ids.insert(override_value.slot_id).second ||
            !authored.contains("function") || !parse_asset_reference(authored["function"], override_value.function))
            return material_instance_authoring_result::failure(
                {.code = material_asset_error_code::invalid_document,
                 .message = "Material Instance contains an invalid or duplicate Function Slot override"});

        const auto& inputs = authored.contains("inputOverrides") ? authored["inputOverrides"] : json::array();
        if (!inputs.is_array())
            return material_instance_authoring_result::failure(
                {.code = material_asset_error_code::invalid_document,
                 .message = "Material Instance Function Slot inputOverrides must be an array"});
        std::set<std::string> pin_ids;
        for (const auto& input : inputs)
        {
            if (!input.is_object() || !input.contains("pinId") || !input["pinId"].is_string() ||
                !input.contains("value"))
                return material_instance_authoring_result::failure(
                    {.code = material_asset_error_code::invalid_document,
                     .message = "Material Instance Function Slot input override is malformed"});
            const auto pin_id = input["pinId"].get<std::string>();
            if (pin_id.empty() || !pin_ids.insert(pin_id).second)
                return material_instance_authoring_result::failure(
                    {.code = material_asset_error_code::invalid_document,
                     .message = "Material Instance contains an invalid or duplicate Function Slot input override"});
            override_value.input_overrides.push_back({.pin_id = pin_id, .value_json = input["value"].dump()});
        }
        result.function_overrides.push_back(std::move(override_value));
    }

    result.version = material_instance_authoring_version;
    result.canonical_json = document.dump();
    return material_instance_authoring_result::success(std::move(result));
}

std::vector<std::byte> serialize_material_instance_package_v1(const material_instance_package_v1& package)
{
    std::vector<std::byte> output;
    append_string(output, material_instance_package_signature);
    append_value(output, material_instance_package_version);
    append_string(output, package.parent_guid);
    append_value(output, package.function_specialization_key);
    const auto material_bytes = serialize_material_package_v4(package.material);
    append_value(output, static_cast<std::uint64_t>(material_bytes.size()));
    output.insert(output.end(), material_bytes.begin(), material_bytes.end());
    append_string(output, package.canonical_instance_json);
    return output;
}

material_instance_package_v1_result deserialize_material_instance_package_v1(std::span<const std::byte> bytes)
{
    package_reader reader(bytes);
    std::string signature;
    std::uint32_t version{};
    material_instance_package_v1 package;
    std::uint64_t material_size{};
    if (!reader.string(signature) || signature != material_instance_package_signature || !reader.value(version) ||
        version != material_instance_package_version || !reader.string(package.parent_guid) ||
        package.parent_guid.empty() || !reader.value(package.function_specialization_key) ||
        !reader.value(material_size) || material_size > bytes.size())
        return material_instance_package_v1_result::failure(
            {.code = material_asset_error_code::corrupt_package,
             .message = "Material Instance package header is invalid"});

    std::vector<std::byte> material_bytes(static_cast<std::size_t>(material_size));
    if (!reader.raw(material_bytes))
        return material_instance_package_v1_result::failure(
            {.code = material_asset_error_code::corrupt_package,
             .message = "Material Instance package contains a truncated Material payload"});
    auto material = deserialize_material_package_v4(material_bytes);
    if (!material) return material_instance_package_v1_result::failure(material.error());
    package.material = std::move(material).value();

    if (!reader.string(package.canonical_instance_json) || !reader.complete())
        return material_instance_package_v1_result::failure(
            {.code = material_asset_error_code::corrupt_package,
             .message = "Material Instance package payload is truncated"});
    auto authored = parse_material_instance_authoring_json(package.canonical_instance_json);
    if (!authored)
        return material_instance_package_v1_result::failure(
            {.code = material_asset_error_code::corrupt_package,
             .message = "Material Instance package contains an invalid authored document"});
    return material_instance_package_v1_result::success(std::move(package));
}

std::vector<std::byte> serialize_material_package_v4(const material_package_v4& package)
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

material_package_v4_result deserialize_material_package_v4(std::span<const std::byte> bytes)
{
    package_reader reader(bytes);
    std::string signature;
    material_package_v4 package;
    if (!reader.string(signature) || signature != material_package_signature ||
        !reader.value(package.compiled.contract_version) || !reader.value(package.compiled.material_abi) ||
        !reader.value(package.compiled.package.high) || !reader.value(package.compiled.package.low))
        return material_package_v4_result::failure(
            {.code = material_asset_error_code::corrupt_package, .message = "Material package header is invalid"});

    if (package.compiled.contract_version != material_pass_contract_version ||
        package.compiled.material_abi != material_abi_version)
        return material_package_v4_result::failure(
            {.code = material_asset_error_code::unsupported_version,
             .message = "Material package uses an unsupported pass contract or Material ABI"});

    std::uint32_t pass_count{};
    if (!reader.value(pass_count) || pass_count > 32u)
        return material_package_v4_result::failure(
            {.code = material_asset_error_code::corrupt_package, .message = "Material package pass table is invalid"});
    if (pass_count != 0 && !package.compiled.package.valid())
        return material_package_v4_result::failure(
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
            return material_package_v4_result::failure({.code = material_asset_error_code::corrupt_package,
                                                        .message = "Material package pass entry is invalid"});
        binding.permutation = {permutation};
        binding.entry_point = {entry_point};
        if (!binding.permutation.valid() || !binding.entry_point.valid() ||
            find_material_pass_binding(package.compiled, binding.pass) != nullptr)
            return material_package_v4_result::failure(
                {.code = material_asset_error_code::corrupt_package,
                 .message = "Material package contains an invalid or duplicate pass binding"});
        package.compiled.passes.push_back(binding);
    }

    std::uint32_t parameter_count{};
    if (!reader.value(parameter_count) || parameter_count > 65'536u)
        return material_package_v4_result::failure(
            {.code = material_asset_error_code::corrupt_package, .message = "Material parameter table is invalid"});
    package.parameters.reserve(parameter_count);
    for (std::uint32_t index = 0; index < parameter_count; ++index)
    {
        shader_parameter_descriptor parameter;
        if (!read_parameter(reader, parameter))
            return material_package_v4_result::failure(
                {.code = material_asset_error_code::corrupt_package, .message = "Material parameter entry is invalid"});
        package.parameters.push_back(std::move(parameter));
    }

    if (!reader.string(package.canonical_document_json) || !reader.complete())
        return material_package_v4_result::failure(
            {.code = material_asset_error_code::corrupt_package, .message = "Material package payload is truncated"});

    if (package.compiled.passes.empty())
    {
        const auto document = json::parse(package.canonical_document_json, nullptr, false);
        if (!document.is_object() || document.value("domain", std::string{"surface"}) != "terrain")
            return material_package_v4_result::failure(
                {.code = material_asset_error_code::corrupt_package,
                 .message = "Surface material package must contain compiled pass bindings"});
    }

    return material_package_v4_result::success(std::move(package));
}

} // namespace arc::render::tools
