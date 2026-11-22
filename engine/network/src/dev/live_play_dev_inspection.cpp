#include <arc/network/live_play_dev_inspection.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <utility>

namespace arc::network::live_play_dev
{
namespace
{
using json = nlohmann::json;
bool valid_token(std::string_view text)
{
    if (text.empty() || text.size() > 128) return false;
    return std::all_of(text.begin(), text.end(), [](unsigned char value)
                       { return std::isalnum(value) != 0 || value == '_' || value == '-' ||
                                value == '.' || value == ':'; });
}
bool valid_guid(std::string_view text)
{
    if (text.size() != 32 && text.size() != 36) return false;
    for (std::size_t i = 0; i < text.size(); ++i)
    {
        const bool dash = text.size() == 36 && (i == 8 || i == 13 || i == 18 || i == 23);
        if (dash)
        {
            if (text[i] != '-') return false;
        }
        else if (std::isxdigit(static_cast<unsigned char>(text[i])) == 0)
            return false;
    }
    return true;
}
} // namespace

std::optional<inspection_request> decode_inspection_request(std::string_view request_id,
                                                             std::string_view payload_json)
{
    if (!valid_token(request_id) || payload_json.size() > 64u * 1024u) return std::nullopt;
    const auto parsed = json::parse(payload_json.begin(), payload_json.end(), nullptr, false);
    if (!parsed.is_object() || parsed.size() != 2 || !parsed.contains("entity") || !parsed.contains("component"))
        return std::nullopt;
    const auto& entity = parsed.at("entity");
    const auto& component = parsed.at("component");
    if (!entity.is_object() || entity.size() != 2 || !entity.contains("kind") ||
        !entity.at("kind").is_string() || !component.is_object() || component.size() != 2 ||
        !component.contains("typeId") || !component.at("typeId").is_string() ||
        !component.contains("schemaVersion") || !component.at("schemaVersion").is_number_unsigned())
        return std::nullopt;
    const auto& kind = entity.at("kind").get_ref<const std::string&>();
    const auto key = kind == "authored" ? "entityGuid" : "runtimeEntityId";
    if ((kind != "authored" && kind != "runtime") || !entity.contains(key) ||
        !entity.at(key).is_string())
        return std::nullopt;
    const auto& entity_id = entity.at(key).get_ref<const std::string&>();
    const auto& component_id = component.at("typeId").get_ref<const std::string&>();
    const auto version = component.at("schemaVersion").get<std::uint64_t>();
    if ((kind == "authored" ? !valid_guid(entity_id) : !valid_token(entity_id)) ||
        !valid_token(component_id) || version == 0 || version > 65535)
        return std::nullopt;
    return inspection_request{.request_id = std::string(request_id),
                              .entity = {.authored = kind == "authored", .id = entity_id},
                              .component = component_id,
                              .schema_version = static_cast<std::uint32_t>(version)};
}

bool inspection_mailbox::enqueue(inspection_request request)
{
    std::scoped_lock lock{mutex_};
    if (requests_.size() >= 64) return false;
    requests_.push_back(std::move(request));
    return true;
}

std::vector<inspection_request> inspection_mailbox::take_requests()
{
    std::scoped_lock lock{mutex_};
    std::vector<inspection_request> output;
    output.reserve(requests_.size());
    while (!requests_.empty())
    {
        output.push_back(std::move(requests_.front()));
        requests_.pop_front();
    }
    return output;
}

bool inspection_mailbox::report(inspection_outcome result)
{
    std::scoped_lock lock{mutex_};
    if (results_.size() >= 64) return false;
    results_.push_back(std::move(result));
    return true;
}

std::vector<inspection_outcome> inspection_mailbox::take_results()
{
    std::scoped_lock lock{mutex_};
    std::vector<inspection_outcome> output;
    output.reserve(results_.size());
    while (!results_.empty())
    {
        output.push_back(std::move(results_.front()));
        results_.pop_front();
    }
    return output;
}

void inspection_mailbox::publish(hierarchy_snapshot snapshot)
{
    std::scoped_lock lock{mutex_};
    snapshot_ = std::move(snapshot);
}

std::optional<hierarchy_snapshot> inspection_mailbox::take_snapshot()
{
    std::scoped_lock lock{mutex_};
    auto output = std::move(snapshot_);
    snapshot_.reset();
    return output;
}

void inspection_mailbox::clear()
{
    std::scoped_lock lock{mutex_};
    requests_.clear();
    results_.clear();
    snapshot_.reset();
}
} // namespace arc::network::live_play_dev
