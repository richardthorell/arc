#include <arc/network/live_play_dev_edits.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace arc::network::live_play_dev
{
namespace
{
using json = nlohmann::json;
constexpr std::uint64_t max_wire_integer = 9007199254740991ULL;

bool exact_keys(const json& object, std::initializer_list<std::string_view> keys)
{
    if (!object.is_object() || object.size() != keys.size()) return false;
    for (const auto name : keys)
        if (!object.contains(std::string(name))) return false;
    return true;
}

bool id_token(std::string_view value)
{
    if (value.empty() || value.size() > 128) return false;
    return std::all_of(value.begin(), value.end(),
                       [](unsigned char c)
                       {
                           return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                                  c == '_' || c == '.' || c == '-' || c == ':';
                       });
}

bool guid(std::string_view value)
{
    if (value.size() != 32 && value.size() != 36) return false;
    for (std::size_t i = 0; i < value.size(); ++i)
    {
        const bool dashed = value.size() == 36 && (i == 8 || i == 13 || i == 18 || i == 23);
        if (dashed)
        {
            if (value[i] != '-') return false;
        }
        else if (!((value[i] >= '0' && value[i] <= '9') || (value[i] >= 'a' && value[i] <= 'f') ||
                   (value[i] >= 'A' && value[i] <= 'F')))
            return false;
    }
    return true;
}

bool bounded_revision(const json& source, const char* key, std::uint64_t& value)
{
    if (!source.contains(key) || !source.at(key).is_number_unsigned()) return false;
    value = source.at(key).get<std::uint64_t>();
    return value <= max_wire_integer;
}
} // namespace

std::optional<property_edit> decode_property_edit(std::string_view request_id, std::string_view payload_json)
{
    if (!id_token(request_id) || payload_json.size() > 64u * 1024u) return std::nullopt;
    const auto payload = json::parse(payload_json.begin(), payload_json.end(), nullptr, false);
    if (!exact_keys(payload, {"entity", "component", "property", "value", "authoringRevision",
                              "expectedRuntimeRevision", "phase"}))
        return std::nullopt;
    const auto& entity = payload.at("entity");
    const auto& component = payload.at("component");
    if (!exact_keys(entity, {"kind", "entityGuid"}) || entity.at("kind") != "authored" ||
        !entity.at("entityGuid").is_string() || !guid(entity.at("entityGuid").get<std::string>()) ||
        !exact_keys(component, {"typeId", "schemaVersion"}) || !component.at("typeId").is_string() ||
        !id_token(component.at("typeId").get<std::string>()) || !component.at("schemaVersion").is_number_unsigned())
        return std::nullopt;
    const auto version = component.at("schemaVersion").get<std::uint64_t>();
    if (version == 0 || version > 65535 || !payload.at("property").is_string() || !payload.at("phase").is_string())
        return std::nullopt;
    property_edit result;
    result.request_id = request_id;
    result.entity_guid = entity.at("entityGuid").get<std::string>();
    result.component = component.at("typeId").get<std::string>();
    result.schema_version = static_cast<std::uint32_t>(version);
    result.property = payload.at("property").get<std::string>();
    result.phase = payload.at("phase").get<std::string>();
    if (!id_token(result.property) ||
        (result.phase != "begin" && result.phase != "update" && result.phase != "commit" && result.phase != "cancel") ||
        !bounded_revision(payload, "authoringRevision", result.authoring_revision) ||
        !bounded_revision(payload, "expectedRuntimeRevision", result.expected_runtime_revision))
        return std::nullopt;
    const auto& values = payload.at("value");
    if (!values.is_array() || values.empty() || values.size() > result.values.size()) return std::nullopt;
    result.value_count = static_cast<std::uint8_t>(values.size());
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (!values[i].is_number()) return std::nullopt;
        const double number = values[i].get<double>();
        if (!std::isfinite(number) || std::abs(number) > 1000000.0 ||
            std::abs(number) > std::numeric_limits<float>::max())
            return std::nullopt;
        result.values[i] = static_cast<float>(number);
    }
    return result;
}

std::optional<property_edit> decode_structural_edit(std::string_view type, std::string_view request_id,
                                                    std::string_view payload_json)
{
    if (!id_token(request_id) || payload_json.size() > 64u * 1024u) return std::nullopt;
    const auto payload = json::parse(payload_json.begin(), payload_json.end(), nullptr, false);
    property_edit result;
    result.request_id = request_id;
    if (type == "entity.create")
    {
        if (!exact_keys(payload, {"entityGuid", "parent", "archetype", "authoringRevision"}) ||
            !payload.at("entityGuid").is_string() || !guid(payload.at("entityGuid").get<std::string>()) ||
            !payload.at("parent").is_null() || payload.at("archetype") != "empty" ||
            !bounded_revision(payload, "authoringRevision", result.authoring_revision) ||
            result.authoring_revision == 0)
            return std::nullopt;
        result.entity_guid = payload.at("entityGuid").get<std::string>();
        result.operation = edit_operation::create;
    }
    else if (type == "entity.delete")
    {
        if (!exact_keys(payload, {"entity", "expectedRuntimeRevision"})) return std::nullopt;
        const auto& entity = payload.at("entity");
        if (!exact_keys(entity, {"kind", "entityGuid"}) || entity.at("kind") != "authored" ||
            !entity.at("entityGuid").is_string() || !guid(entity.at("entityGuid").get<std::string>()) ||
            !bounded_revision(payload, "expectedRuntimeRevision", result.expected_runtime_revision))
            return std::nullopt;
        result.entity_guid = entity.at("entityGuid").get<std::string>();
        result.operation = edit_operation::erase;
    }
    else
        return std::nullopt;
    return result;
}

bool edit_mailbox::enqueue(property_edit edit)
{
    std::scoped_lock lock{mutex_};
    if (pending_.size() >= maximum_pending_edits) return false;
    pending_.push_back(std::move(edit));
    return true;
}

std::vector<property_edit> edit_mailbox::take_edits()
{
    std::scoped_lock lock{mutex_};
    std::vector<property_edit> result;
    result.reserve(pending_.size());
    while (!pending_.empty())
    {
        result.push_back(std::move(pending_.front()));
        pending_.pop_front();
    }
    return result;
}

bool edit_mailbox::report(edit_result result)
{
    std::scoped_lock lock{mutex_};
    if (results_.size() >= maximum_results) return false;
    results_.push_back(std::move(result));
    return true;
}

std::vector<edit_result> edit_mailbox::take_results()
{
    std::scoped_lock lock{mutex_};
    std::vector<edit_result> result;
    result.reserve(results_.size());
    while (!results_.empty())
    {
        result.push_back(std::move(results_.front()));
        results_.pop_front();
    }
    return result;
}

void edit_mailbox::clear()
{
    std::scoped_lock lock{mutex_};
    pending_.clear();
    results_.clear();
}

} // namespace arc::network::live_play_dev
