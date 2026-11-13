#include <arc/render/cooked_material_instance.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <optional>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string_view>
#include <utility>

namespace arc::render
{
namespace
{
class reader
{
public:
    explicit reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <typename T> bool value(T& output)
    {
        if (offset_ > bytes_.size() || sizeof(T) > bytes_.size() - offset_) return false;
        std::memcpy(&output, bytes_.data() + offset_, sizeof(T));
        offset_ += sizeof(T);
        return true;
    }

    bool string(std::string& output)
    {
        std::uint64_t size{};
        if (!value(size) || size > 8u * 1024u * 1024u || size > bytes_.size() - offset_) return false;
        output.assign(reinterpret_cast<const char*>(bytes_.data() + offset_), static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
        return true;
    }

    bool slice(std::size_t size, std::span<const std::byte>& output)
    {
        if (size > bytes_.size() - offset_) return false;
        output = bytes_.subspan(offset_, size);
        offset_ += size;
        return true;
    }

    [[nodiscard]] bool complete() const noexcept { return offset_ == bytes_.size(); }

private:
    std::span<const std::byte> bytes_;
    std::size_t offset_{};
};

std::optional<shader_parameter_id> parse_parameter(std::string_view text)
{
    int base = 10;
    if (text.starts_with("0x") || text.starts_with("0X"))
    {
        text.remove_prefix(2);
        base = 16;
    }
    std::uint64_t value{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value == 0) return {};
    return shader_parameter_id{value};
}

using json = nlohmann::json;

bool finite_number(const json& value)
{
    return value.is_number() && std::isfinite(value.get<double>());
}

core::result<material_parameter_value, std::string> typed_value(const json& value, shader_parameter_type type)
{
    using result = core::result<material_parameter_value, std::string>;
    if (type == shader_parameter_type::boolean && value.is_boolean())
        return result::success(material_parameter_value{value.get<bool>()});
    if (type == shader_parameter_type::int32 && value.is_number_integer())
    {
        const auto number = value.get<std::int64_t>();
        if (number >= std::numeric_limits<std::int32_t>::min() &&
            number <= std::numeric_limits<std::int32_t>::max())
            return result::success(material_parameter_value{static_cast<std::int32_t>(number)});
    }
    if (type == shader_parameter_type::uint32 && value.is_number_unsigned())
    {
        const auto number = value.get<std::uint64_t>();
        if (number <= std::numeric_limits<std::uint32_t>::max())
            return result::success(material_parameter_value{static_cast<std::uint32_t>(number)});
    }
    if (type == shader_parameter_type::float32 && finite_number(value))
        return result::success(material_parameter_value{value.get<float>()});

    const auto vector_size = type == shader_parameter_type::float2 ? 2u
                             : type == shader_parameter_type::float3 ? 3u
                             : type == shader_parameter_type::float4 ? 4u
                                                                      : 0u;
    if (vector_size > 0 && value.is_array() && value.size() == vector_size)
    {
        for (const auto& component : value)
            if (!finite_number(component)) return result::failure("Invalid nonfinite vector component");
        if (vector_size == 2)
        {
            math::vector2f vector{};
            for (std::size_t i = 0; i < 2; ++i) vector[i] = value[i].get<float>();
            return result::success(material_parameter_value{vector});
        }
        if (vector_size == 3)
        {
            math::vector3f vector{};
            for (std::size_t i = 0; i < 3; ++i) vector[i] = value[i].get<float>();
            return result::success(material_parameter_value{vector});
        }
        math::vector4f vector{};
        for (std::size_t i = 0; i < 4; ++i) vector[i] = value[i].get<float>();
        return result::success(material_parameter_value{vector});
    }
    return result::failure("Unsupported or incompatible cooked Material Instance parameter value");
}
} // namespace

core::result<cooked_material_instance_values, std::string>
decode_cooked_material_instance_values(std::span<const std::byte> bytes)
{
    using result = core::result<cooked_material_instance_values, std::string>;
    reader input(bytes);
    std::string signature;
    std::uint32_t version{};
    std::string parent_name;
    std::uint64_t specialization{}, material_size{};
    std::span<const std::byte> material_bytes;
    std::string document;
    if (!input.string(signature) || signature != "ARC_MATERIAL_INSTANCE_1" || !input.value(version) || version != 1 ||
        !input.string(parent_name) || !input.value(specialization) || !input.value(material_size) ||
        material_size > bytes.size() || !input.slice(static_cast<std::size_t>(material_size), material_bytes) ||
        !input.string(document) || !input.complete())
        return result::failure("Invalid or truncated cooked Material Instance envelope");
    if (specialization != 0)
        return result::failure("Cooked Function Slot specialization requires its own runtime shader artifacts");
    const auto parent = assets::parse_asset_guid(parent_name);
    if (!parent) return result::failure("Material Instance parent GUID is invalid");
    auto metadata = decode_cooked_material_metadata(material_bytes);
    if (!metadata) return result::failure("Material Instance parent package: " + metadata.error());
    if (metadata.value().compiled.package.high != parent->high ||
        metadata.value().compiled.package.low != parent->low)
        return result::failure("Material Instance embedded parent program identity mismatch");

    auto authored = json::parse(document, nullptr, false);
    if (authored.is_discarded() || !authored.is_object() || !authored.contains("version") ||
        !authored["version"].is_number_integer() || authored["version"] != 1 ||
        !authored.contains("parent") || !authored["parent"].is_object() ||
        !authored["parent"].contains("guid") || !authored["parent"]["guid"].is_string())
        return result::failure("Material Instance authoring document is invalid");
    const auto authored_parent = assets::parse_asset_guid(authored["parent"]["guid"].get<std::string>());
    if (!authored_parent || *authored_parent != *parent)
        return result::failure("Material Instance authored parent differs from cooked parent");
    if (!authored.contains("name") || !authored["name"].is_string() || authored["name"].get<std::string>().empty())
        return result::failure("Material Instance requires an authored name");
    if (authored.contains("functionOverrides") &&
        (!authored["functionOverrides"].is_array() || !authored["functionOverrides"].empty()))
        return result::failure("Material Instance Function Slot overrides are not supported by this runtime path");

    cooked_material_instance_values instance;
    instance.parent = *parent;
    instance.parent_material = std::move(metadata).value();
    instance.name = authored["name"].get<std::string>();
    const auto overrides = authored.value("parameterOverrides", json::array());
    if (!overrides.is_array() || overrides.size() > 1024)
        return result::failure("Invalid Material Instance parameter override list");
    std::set<std::uint64_t> seen;
    for (const auto& item : overrides)
    {
        if (!item.is_object() || !item.contains("parameterId") || !item["parameterId"].is_string() ||
            !item.contains("value"))
            return result::failure("Malformed Material Instance parameter override");
        const auto id = parse_parameter(item["parameterId"].get<std::string>());
        if (!id || !seen.insert(id->representation()).second)
            return result::failure("Duplicate or invalid Material Instance parameter identity");
        const auto& parameters = instance.parent_material.parameters;
        const auto found = std::find_if(parameters.begin(), parameters.end(),
                                        [&](const auto& parameter) { return parameter.id == *id; });
        if (found == parameters.end()) return result::failure("Material Instance parameter absent from its parent");
        auto parsed = typed_value(item["value"], found->type);
        if (!parsed) return result::failure(parsed.error());
        if (found->has_range && found->type == shader_parameter_type::float32)
        {
            const auto scalar = std::get<float>(parsed.value());
            if (scalar < found->minimum || scalar > found->maximum)
                return result::failure("Material Instance scalar override outside authored range");
        }
        instance.overrides.push_back({.id = *id, .name = found->name, .value = std::move(parsed).value()});
    }
    return result::success(std::move(instance));
}

core::result<material_descriptor, std::string>
apply_cooked_material_instance_values(material_descriptor parent, const cooked_material_instance_values& instance)
{
    using result = core::result<material_descriptor, std::string>;
    if (!parent.runtime_program || parent.shader_package.high != instance.parent.high ||
        parent.shader_package.low != instance.parent.low)
        return result::failure("Material Instance parent runtime program identity mismatch");
    parent.name = instance.name;
    parent.parameters = instance.overrides;
    return result::success(std::move(parent));
}
} // namespace arc::render
