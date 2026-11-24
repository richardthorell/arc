#pragma once

/** @file
 * Optional Development-only native client for the ARC Editor Live Play control
 * plane. Shipping/gameplay networking never links this library.
 *
 * The launcher hands us a single protected anonymous pipe/handle carrying the
 * entire session configuration. Never put its secret in argv, URLs, environment,
 * log output, cooked assets, or the packaged distribution manifest.
 */

#include <arc/network/live_play_dev_edits.h>
#include <arc/network/live_play_dev_inspection.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace arc::network::live_play_dev
{

inline constexpr std::uint32_t protocol_version = 1;
inline constexpr std::uint32_t maximum_frame_bytes = 64u * 1024u;
inline constexpr std::size_t secret_bytes = 32;
inline constexpr std::uint32_t maximum_handoff_bytes = 4096;

/** Session-private C9 release pairing, never serialized to the public D0 or CLI. */
struct content_handoff
{
    std::string origin;
    std::string token; // 64 lowercase hex bytes, erased on destruction.
    std::string scene_guid;
    std::string scene_asset_guid;
    std::string authoring_sha256;
    std::string descriptor_sha256;
    std::string texture_family;
    std::uint64_t authoring_revision{};
    std::uint64_t generation{};

    ~content_handoff() noexcept;
};

struct launch_handoff
{
    std::uint16_t port{};
    std::array<std::uint8_t, secret_bytes> secret{};
    std::string project_guid;
    std::string build_id;
    std::string target_profile;
    std::string session_id;
    std::string instance_id;
    std::string schema_hash;
    std::vector<std::string> capabilities;
    std::optional<content_handoff> content;

    ~launch_handoff() noexcept;
};

/**
 * Consume the one-time launch record from an OS-inherited, read-only pipe.
 * \p handle is a Windows HANDLE or an Android/POSIX file descriptor, cast to
 * uintptr_t by the trusted platform bootstrap. The caller owns closing it.
 *
 * Record: uint32 big-endian byte length, then bounded JSON with a 64-hex-digit
 * secret. The pipe must be inherited or passed through privileged launcher IPC.
 * No disk or command-line fallback is provided.
 */
[[nodiscard]] bool read_launch_handoff(std::uintptr_t handle, launch_handoff& result, std::string& error);

/** Fully validated cooked-baseline offer from an authenticated editor. */
struct offered_baseline
{
    std::string scene_guid;
    std::string authoring_sha256;
    std::string cooked_release_sha256;
    std::uint64_t authoring_revision{};
    std::uint64_t cooked_generation{};
};

struct incoming_message
{
    std::string type;
    std::string message_id;
    std::uint64_t sequence{};
    std::string json;                  ///< Strictly bounded, validated wire envelope. Never blindly execute it.
    std::optional<property_edit> edit; ///< Parsed typed edit, never raw ECS mutation.
    std::optional<inspection_request> inspection; ///< Read-only ECS request.
    std::optional<offered_baseline> baseline;     ///< Validated offer identity, no raw JSON required in host.
};

/**
 * Synchronous nonblocking-poll client: connect/handshake at startup and call
 * poll() from the host's chosen safe execution boundary. No socket callbacks
 * ever mutate ECS, assets, render state or authoring state.
 */
class client final
{
public:
    client() = default;
    ~client() noexcept;

    client(const client&) = delete;
    client& operator=(const client&) = delete;
    client(client&&) = delete;
    client& operator=(client&&) = delete;

    [[nodiscard]] bool connect(const launch_handoff& handoff, std::string& error);
    /** Reauthenticate the same native instance using only the inherited in-memory handoff. */
    [[nodiscard]] bool reconnect(const launch_handoff& handoff, std::string& error);
    [[nodiscard]] bool connected() const noexcept;
    void close() noexcept;

    [[nodiscard]] bool send_state(std::string_view state, std::uint64_t tick, std::uint64_t revision,
                                  std::string& error);
    [[nodiscard]] bool send_heartbeat(std::string& error);
    [[nodiscard]] bool send_ack(std::string_view request_id, std::uint64_t applied_revision, std::string& error);
    [[nodiscard]] bool send_rejected_ack(std::string_view request_id, std::string_view code, std::string& error);
    [[nodiscard]] bool send_content_generation(std::uint64_t generation, std::string_view status, std::string& error);
    [[nodiscard]] bool send_hierarchy_snapshot(const hierarchy_snapshot& snapshot, std::string& error);
    [[nodiscard]] bool send_hierarchy_changed(const hierarchy_delta& delta, std::string& error);
    [[nodiscard]] bool send_inspection_result(const component_inspection& result, std::string& error);

    /**
     * Call ONLY after the C9 loader has verified and activated or rejected
     * the offered cooked release. Polling an offer never activates a scene.
     * Rejection reason: unavailable/incompatible/corrupt/unsupported.
     */
    [[nodiscard]] bool send_baseline_status(std::string_view status, std::string_view reason, std::string& error);

    /** Empty optional means no frame during the timeout, not a command. */
    [[nodiscard]] std::optional<incoming_message> poll(std::uint32_t timeout_ms, std::string& error);
    [[nodiscard]] std::uint64_t last_received_sequence() const noexcept;

private:
    [[nodiscard]] bool connect_impl(const launch_handoff& handoff, std::string& error, bool resume);
    [[nodiscard]] bool send_message(std::string_view type, const std::string& payload_json, std::string& error);
    [[nodiscard]] bool receive_frame(std::string& json, std::uint32_t timeout_ms, std::string& error);

    std::intptr_t socket_{-1};
    std::uint64_t outbound_sequence_{};
    std::uint64_t inbound_sequence_{};
    // Monotonic across transient socket failures; reset only by a fresh launch.
    std::uint64_t resume_outbound_sequence_{};
    std::uint64_t resume_inbound_sequence_{};
    std::string project_guid_;
    std::string build_id_;
    std::string target_profile_;
    std::string session_id_;
    std::string instance_id_;
    std::vector<std::string> negotiated_capabilities_;
    std::string pending_baseline_request_;
    std::string pending_baseline_scene_;
    std::string pending_baseline_release_;
    std::uint64_t pending_baseline_generation_{};
    bool initialized_socket_api_{};
};

} // namespace arc::network::live_play_dev
