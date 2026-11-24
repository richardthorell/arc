#include "android_live_play_pairing.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace arc::platform::android
{
namespace
{
constexpr std::size_t maximum_pairing_bytes = 2048;
constexpr std::string_view pairing_file = ".arc-live-play-pairing.json";
using json = nlohmann::json;

bool ascii_token(std::string_view value, std::size_t min, std::size_t max) noexcept
{
    return value.size() >= min && value.size() <= max &&
           std::all_of(value.begin(), value.end(), [](unsigned char c)
                       { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9') || c == '_' || c == '-'; });
}

bool hex_lower(std::string_view value) noexcept
{
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](unsigned char c)
                                            { return (c >= 'a' && c <= 'f') || (c >= '0' && c <= '9'); });
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
        else if (!(value[i] >= 'a' && value[i] <= 'f') && !(value[i] >= '0' && value[i] <= '9'))
            return false;
    }
    return true;
}

bool profile(std::string_view value) noexcept
{
    if (value.empty() || value.size() > 128) return false;
    if (!(value[0] >= 'A' && value[0] <= 'Z') && !(value[0] >= 'a' && value[0] <= 'z') &&
        !(value[0] >= '0' && value[0] <= '9'))
        return false;
    return std::all_of(value.begin(), value.end(), [](unsigned char c)
                       { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                (c >= '0' && c <= '9') || c == '_' || c == '-'; });
}

bool loopback_origin(std::string_view origin) noexcept
{
    constexpr std::string_view prefix = "http://127.0.0.1:";
    if (!origin.starts_with(prefix)) return false;
    auto digits = origin.substr(prefix.size());
    if (digits.empty() || digits.size() > 5 || digits.front() == '0') return false;
    std::uint32_t port = 0;
    for (const auto ch : digits)
    {
        if (ch < '0' || ch > '9') return false;
        port = port * 10 + static_cast<std::uint32_t>(ch - '0');
    }
    return port > 0 && port <= 65535;
}

std::optional<std::uint64_t> positive_integer(const json& value)
{
    if (!value.is_number_unsigned() && !value.is_number_integer()) return std::nullopt;
    if (value.is_number_integer() && !value.is_number_unsigned() && value.get<std::int64_t>() < 0)
        return std::nullopt;
    return value.get<std::uint64_t>();
}

std::optional<android_live_play_pairing> parse_pairing(std::string_view bytes)
{
    auto record = json::parse(bytes.begin(), bytes.end(), nullptr, false);
    if (!record.is_object() || record.size() != 13 || !record.contains("format") ||
        !record["format"].is_string() ||
        record["format"].get<std::string>() != "arc.android-live-play-pairing" ||
        !record.contains("version") || positive_integer(record["version"]) != 1)
        return std::nullopt;

    constexpr std::string_view strings[] = {"projectGuid", "buildId", "targetProfile", "textureFamily",
                                            "sessionId",   "instanceId", "descriptorSha256", "origin", "token"};
    for (const auto name : strings)
    {
        if (!record.contains(std::string(name)) || !record[std::string(name)].is_string())
            return std::nullopt;
    }
    if (!record.contains("generation") || !record.contains("expiresAtEpochMs")) return std::nullopt;
    const auto generation = positive_integer(record["generation"]);
    const auto expiry = positive_integer(record["expiresAtEpochMs"]);
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    if (!generation || !*generation || !expiry || *expiry <= static_cast<std::uint64_t>(now) ||
        *expiry > static_cast<std::uint64_t>(now) + 300000)
        return std::nullopt;

    android_live_play_pairing pairing;
    pairing.project_guid = record["projectGuid"].get<std::string>();
    pairing.build_id = record["buildId"].get<std::string>();
    pairing.target_profile = record["targetProfile"].get<std::string>();
    pairing.texture_family = record["textureFamily"].get<std::string>();
    pairing.session_id = record["sessionId"].get<std::string>();
    pairing.instance_id = record["instanceId"].get<std::string>();
    pairing.descriptor_sha256 = record["descriptorSha256"].get<std::string>();
    pairing.generation = *generation;
    pairing.origin = record["origin"].get<std::string>();
    pairing.token = record["token"].get<std::string>();
    if (!guid(pairing.project_guid) || !ascii_token(pairing.build_id, 1, 128) ||
        !profile(pairing.target_profile) || !ascii_token(pairing.texture_family, 1, 32) ||
        !ascii_token(pairing.session_id, 1, 128) || !ascii_token(pairing.instance_id, 1, 128) ||
        !hex_lower(pairing.descriptor_sha256) || !hex_lower(pairing.token) || !loopback_origin(pairing.origin))
        return std::nullopt;
    // The Android host has no source-controlled path or untrusted Intent secret.
    // Only the private app process receives this exact, short-lived identity.
    return pairing;
}
} // namespace

android_live_play_pairing::~android_live_play_pairing() noexcept
{
    std::fill(token.begin(), token.end(), '\0');
}

android_live_play_pairing_result consume_android_live_play_pairing(const char* internal_data_path)
{
    if (!internal_data_path || !*internal_data_path) return {};
    const auto file = std::string(internal_data_path) + "/" + std::string(pairing_file);
    const int descriptor = ::open(file.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor == -1) return {.present = errno != ENOENT, .pairing = std::nullopt};

    struct stat info{};
    const bool accepted = ::fstat(descriptor, &info) == 0 && S_ISREG(info.st_mode) &&
                          (info.st_mode & 0777) == 0600 && info.st_uid == ::geteuid() &&
                          info.st_nlink == 1 && info.st_size > 0 &&
                          static_cast<std::uint64_t>(info.st_size) <= maximum_pairing_bytes;
    std::string bytes;
    bool read_ok = accepted;
    if (accepted)
    {
        bytes.resize(static_cast<std::size_t>(info.st_size));
        std::size_t offset = 0;
        while (offset < bytes.size())
        {
            const auto count = ::read(descriptor, bytes.data() + offset, bytes.size() - offset);
            if (count <= 0)
            {
                read_ok = false;
                break;
            }
            offset += static_cast<std::size_t>(count);
        }
    }
    // Unlink even on rejection. A crash or restart must never replay an old
    // protected session; the editor stages a fresh record for each Play.
    const bool unlinked = ::unlink(file.c_str()) == 0;
    ::close(descriptor);
    if (!read_ok || !unlinked) return {.present = true, .pairing = std::nullopt};
    auto pairing = parse_pairing(bytes);
    std::fill(bytes.begin(), bytes.end(), '\0');
    return {.present = true, .pairing = std::move(pairing)};
}

} // namespace arc::platform::android
