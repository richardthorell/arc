#include <arc/assets/remote_content.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>

namespace arc::assets
{
namespace
{

using json = nlohmann::json;
constexpr std::size_t max_descriptor_bytes = 8u * 1024u * 1024u;
constexpr std::size_t max_chunks = 100000;
constexpr std::uint64_t max_object_bytes = 1ull << 44u;

bool ascii_hex_lower(std::string_view value) noexcept
{
    return std::all_of(value.begin(), value.end(),
                       [](unsigned char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

bool sha256(std::string_view value) noexcept
{
    return value.size() == 64 && ascii_hex_lower(value);
}

bool guid(std::string_view value) noexcept
{
    if (value.size() != 36) return false;
    for (std::size_t i = 0; i < value.size(); ++i)
    {
        if (i == 8 || i == 13 || i == 18 || i == 23)
        {
            if (value[i] != '-') return false;
        }
        else if (!ascii_hex_lower(value.substr(i, 1)))
            return false;
    }
    return true;
}

bool safe_name(std::string_view value) noexcept
{
    if (value.empty() || value == "." || value == ".." || value.size() > 255) return false;
    const auto alnum = [](unsigned char c)
    { return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
    if (!alnum(static_cast<unsigned char>(value.front()))) return false;
    return std::all_of(value.begin(), value.end(),
                       [&](unsigned char c) { return alnum(c) || c == '.' || c == '_' || c == '-'; });
}

bool suffix(std::string_view value, std::string_view ending) noexcept
{
    return value.size() >= ending.size() && value.substr(value.size() - ending.size()) == ending;
}

std::optional<std::string> string_field(const json& object, std::string_view key)
{
    if (!object.is_object()) return std::nullopt;
    auto it = object.find(std::string(key));
    if (it == object.end() || !it->is_string()) return std::nullopt;
    return it->get<std::string>();
}

std::optional<std::uint64_t> unsigned_field(const json& object, std::string_view key)
{
    if (!object.is_object()) return std::nullopt;
    const auto it = object.find(std::string(key));
    if (it == object.end() || !(it->is_number_unsigned() || it->is_number_integer())) return std::nullopt;
    if (!it->is_number_unsigned() && it->get<std::int64_t>() < 0) return std::nullopt;
    return it->get<std::uint64_t>();
}

std::optional<remote_content_object> parse_object(const json& object)
{
    if (!object.is_object()) return std::nullopt;
    auto digest = string_field(object, "sha256");
    auto path = string_field(object, "path");
    auto size = unsigned_field(object, "size");
    if (!digest || !path || !size || *size == 0 || *size > max_object_bytes || !sha256(*digest) ||
        *path != "objects/sha256/" + *digest)
        return std::nullopt;
    return remote_content_object{.path = std::move(*path), .sha256 = std::move(*digest), .size = *size};
}

bool matches_target(const json& object, const remote_content_target& expected)
{
    if (!object.is_object()) return false;
    constexpr std::array keys{"platform", "architecture", "renderer", "cookProfile", "textureFamily"};
    const std::array values{std::string_view(expected.platform), std::string_view(expected.architecture),
                            std::string_view(expected.renderer), std::string_view(expected.cook_profile),
                            std::string_view(expected.texture_family)};
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        const auto field = string_field(object, keys[i]);
        if (!field || field->empty() || *field != values[i]) return false;
    }
    return true;
}

remote_content_parse_result failure(std::string message)
{
    return {.release = std::nullopt, .error = std::move(message)};
}

} // namespace

remote_content_parse_result parse_remote_content_release(std::string_view bytes,
                                                         const remote_content_expectations& expected)
{
    if (bytes.empty() || bytes.size() > max_descriptor_bytes)
        return failure("release descriptor exceeds metadata limits");
    const auto document = json::parse(bytes.begin(), bytes.end(), nullptr, false);
    if (!document.is_object()) return failure("release descriptor must be valid JSON object");

    const auto format = string_field(document, "format");
    const auto version = unsigned_field(document, "formatVersion");
    if (!format || *format != "arc.remote-content" || !version || *version != 1)
        return failure("unsupported remote release format/version");

    const auto project_guid = string_field(document, "projectGuid");
    const auto engine_version = string_field(document, "engineVersion");
    const auto base_build_id = string_field(document, "baseBuildId");
    if (!project_guid || !guid(*project_guid) || *project_guid != expected.project_guid)
        return failure("incompatible projectGuid");
    if (!engine_version || engine_version->empty() || *engine_version != expected.engine_version)
        return failure("incompatible engineVersion");
    if (!base_build_id || base_build_id->empty() || *base_build_id != expected.base_build_id)
        return failure("incompatible baseBuildId");

    const auto target = document.find("target");
    if (target == document.end() || !matches_target(*target, expected.target))
        return failure("incompatible remote release target");

    const auto generation = unsigned_field(document, "generation");
    if (!generation || *generation == 0 || *generation <= expected.minimum_generation)
        return failure("stale or invalid release generation");
    const auto complete = document.find("complete");
    if (complete == document.end() || !complete->is_boolean() || !complete->get<bool>())
        return failure("remote release generation is incomplete");

    const auto manifest_name = string_field(document, "cookManifestName");
    if (!manifest_name || !safe_name(*manifest_name) || !suffix(*manifest_name, ".arccookmanifest"))
        return failure("invalid cookManifestName");
    const auto manifest_field = document.find("cookManifest");
    const auto manifest = manifest_field == document.end() ? std::nullopt : parse_object(*manifest_field);
    if (!manifest) return failure("invalid cookManifest immutable object");

    const auto chunks = document.find("chunks");
    if (chunks == document.end() || !chunks->is_array() || chunks->size() > max_chunks)
        return failure("remote chunks must be a bounded array");

    remote_content_release release{.project_guid = *project_guid,
                                   .engine_version = *engine_version,
                                   .base_build_id = *base_build_id,
                                   .target = expected.target,
                                   .generation = *generation,
                                   .cook_manifest_name = *manifest_name,
                                   .cook_manifest = *manifest};
    std::set<std::string> names;
    std::set<std::string> hashes{manifest->sha256};
    for (const auto& chunk : *chunks)
    {
        const auto name = string_field(chunk, "name");
        const auto object = parse_object(chunk);
        if (!name || !safe_name(*name) || !suffix(*name, ".arcpak") || !object)
            return failure("invalid cooked chunk name or immutable object");
        auto folded = *name;
        std::transform(folded.begin(), folded.end(), folded.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (!names.insert(std::move(folded)).second || !hashes.insert(object->sha256).second)
            return failure("duplicate cooked chunk name or hash");
        release.chunks.push_back({.name = *name, .object = *object});
    }
    std::sort(release.chunks.begin(), release.chunks.end(),
              [](const auto& a, const auto& b) { return a.name < b.name; });
    return {.release = std::move(release), .error = {}};
}

remote_content_feed_action remote_content_reconnect(std::uint64_t last_acknowledged, std::uint64_t oldest_available,
                                                    std::uint64_t newest_available) noexcept
{
    if (newest_available < oldest_available) return remote_content_feed_action::invalid;
    if (last_acknowledged > newest_available) return remote_content_feed_action::reject_regression;
    if (last_acknowledged == newest_available) return remote_content_feed_action::up_to_date;
    if (oldest_available != 0 && last_acknowledged < oldest_available - 1)
        return remote_content_feed_action::full_snapshot;
    return remote_content_feed_action::replay;
}

} // namespace arc::assets
