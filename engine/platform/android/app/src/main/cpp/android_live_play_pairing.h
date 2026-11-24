#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace arc::platform::android
{

/** Private Development pairing, never Intent extras, command-line args or logcat. */
struct android_live_play_pairing
{
    std::string project_guid;
    std::string build_id;
    std::string target_profile;
    std::string texture_family;
    std::string session_id;
    std::string instance_id;
    std::string descriptor_sha256;
    std::uint64_t generation{};
    std::string origin;
    std::string token;

    ~android_live_play_pairing() noexcept;
};

struct android_live_play_pairing_result
{
    bool present{};
    std::optional<android_live_play_pairing> pairing;

    [[nodiscard]] bool valid() const noexcept
    {
        return !present || pairing.has_value();
    }
};

/** Read and unlink an app-private file, at most once per Activity startup.
 * Reject symlinks, wrong owner/mode, oversized data, expired records,
 * non-loopback origins, and malformed credentials. No diagnostics include
 * record bytes. No source project or external files are read.
 */
[[nodiscard]] android_live_play_pairing_result consume_android_live_play_pairing(const char* internal_data_path);

} // namespace arc::platform::android
