#include <arc/network/live_play_dev_client.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace arc::network::live_play_dev
{
namespace
{
using json = nlohmann::json;
constexpr std::uint64_t maximum_js_safe_integer = 9007199254740991ULL;
constexpr auto read_timeout_ms = 5000u;

#if defined(_WIN32)
using socket_type = SOCKET;
constexpr auto invalid_socket = INVALID_SOCKET;
socket_type native_socket(std::intptr_t value)
{
    return static_cast<socket_type>(value);
}
void close_socket(socket_type s)
{
    if (s != invalid_socket) closesocket(s);
}
#else
using socket_type = int;
constexpr auto invalid_socket = -1;
socket_type native_socket(std::intptr_t value)
{
    return static_cast<socket_type>(value);
}
void close_socket(socket_type s)
{
    if (s != invalid_socket) ::close(s);
}
#endif

void erase_bytes(void* memory, std::size_t bytes) noexcept
{
    volatile auto* pointer = static_cast<volatile std::uint8_t*>(memory);
    while (bytes--)
        *pointer++ = 0;
}

bool safe_number(const json& object, std::string_view key, std::uint64_t& out)
{
    const auto it = object.find(std::string(key));
    if (it == object.end() || !it->is_number_unsigned()) return false;
    out = it->get<std::uint64_t>();
    return out <= maximum_js_safe_integer;
}

bool token(std::string_view value)
{
    if (value.empty() || value.size() > 128) return false;
    return std::all_of(value.begin(), value.end(),
                       [](unsigned char c) { return std::isalnum(c) || c == '.' || c == '_' || c == ':' || c == '-'; });
}

bool guid(std::string_view value)
{
    if (value.size() != 32 && value.size() != 36) return false;
    for (std::size_t i = 0; i < value.size(); ++i)
    {
        const bool dash = value.size() == 36 && (i == 8 || i == 13 || i == 18 || i == 23);
        if (dash ? value[i] != '-' : !std::isxdigit(static_cast<unsigned char>(value[i]))) return false;
    }
    return true;
}

bool hash(std::string_view value)
{
    return value.size() == 64 &&
           std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isxdigit(c) != 0; });
}

bool field(const json& object, const char* key, std::string& out)
{
    const auto it = object.find(key);
    if (it == object.end() || !it->is_string()) return false;
    out = it->get<std::string>();
    return true;
}

bool exact_keys(const json& object, std::initializer_list<std::string_view> fields)
{
    if (!object.is_object() || object.size() != fields.size()) return false;
    for (auto name : fields)
        if (!object.contains(std::string(name))) return false;
    return true;
}

bool known_capability(std::string_view value)
{
    return value == "session.observe" || value == "session.control" || value == "entity.inspect" ||
           value == "entity.edit" || value == "content.observe";
}

bool parse_capabilities(const json& raw, std::vector<std::string>& output)
{
    if (!raw.is_array() || raw.size() > 5) return false;
    std::set<std::string> unique;
    for (const auto& item : raw)
    {
        if (!item.is_string()) return false;
        auto name = item.get<std::string>();
        if (!known_capability(name) || !unique.insert(name).second) return false;
        output.push_back(std::move(name));
    }
    return true;
}

bool read_pipe(std::uintptr_t handle, void* data, std::size_t length)
{
    auto* target = static_cast<std::uint8_t*>(data);
    while (length)
    {
#if defined(_WIN32)
        DWORD amount = 0;
        if (!ReadFile(reinterpret_cast<HANDLE>(handle), target, static_cast<DWORD>(length), &amount, nullptr) ||
            !amount)
            return false;
#else
        const auto amount = ::read(static_cast<int>(handle), target, length);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) return false;
#endif
        target += amount;
        length -= static_cast<std::size_t>(amount);
    }
    return true;
}

bool read_exact(socket_type s, void* destination, std::size_t bytes, std::uint32_t timeout_ms)
{
    auto* output = static_cast<char*>(destination);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (bytes)
    {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) return false;
        const auto remain = std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);
        timeval timeout{static_cast<long>(remain.count() / 1000000), static_cast<long>(remain.count() % 1000000)};
        fd_set readable;
        FD_ZERO(&readable);
        FD_SET(s, &readable);
#if defined(_WIN32)
        const int ready = ::select(0, &readable, nullptr, nullptr, &timeout);
#else
        const int ready = ::select(s + 1, &readable, nullptr, nullptr, &timeout);
#endif
        if (ready <= 0) return false;
        const auto chunk = static_cast<int>(std::min<std::size_t>(bytes, 65536));
        const int received = ::recv(s, output, chunk, 0);
        if (received <= 0) return false;
        output += received;
        bytes -= static_cast<std::size_t>(received);
    }
    return true;
}

bool send_exact(socket_type s, const void* input, std::size_t bytes)
{
    auto* data = static_cast<const char*>(input);
    while (bytes)
    {
        const auto chunk = static_cast<int>(std::min<std::size_t>(bytes, 65536));
#if defined(MSG_NOSIGNAL)
        // A disconnected editor must not terminate the game process via SIGPIPE.
        const int sent = ::send(s, data, chunk, MSG_NOSIGNAL);
#else
        const int sent = ::send(s, data, chunk, 0);
#endif
        if (sent <= 0) return false;
        data += sent;
        bytes -= static_cast<std::size_t>(sent);
    }
    return true;
}

bool send_frame(socket_type s, const void* input, std::size_t bytes)
{
    if (bytes == 0 || bytes > maximum_frame_bytes) return false;
    const std::array<std::uint8_t, 4> header{
        static_cast<std::uint8_t>((bytes >> 24) & 255), static_cast<std::uint8_t>((bytes >> 16) & 255),
        static_cast<std::uint8_t>((bytes >> 8) & 255), static_cast<std::uint8_t>(bytes & 255)};
    return send_exact(s, header.data(), header.size()) && send_exact(s, input, bytes);
}

bool read_frame(socket_type s, std::string& content, std::uint32_t timeout_ms)
{
    std::array<std::uint8_t, 4> header{};
    if (!read_exact(s, header.data(), header.size(), timeout_ms)) return false;
    const std::uint32_t bytes = (std::uint32_t(header[0]) << 24) | (std::uint32_t(header[1]) << 16) |
                                (std::uint32_t(header[2]) << 8) | std::uint32_t(header[3]);
    if (!bytes || bytes > maximum_frame_bytes) return false;
    content.resize(bytes);
    return read_exact(s, content.data(), bytes, timeout_ms);
}

bool valid_authority(const launch_handoff& handoff)
{
    if (!handoff.port || !guid(handoff.project_guid) || !token(handoff.build_id) || !token(handoff.target_profile) ||
        !token(handoff.session_id) || !token(handoff.instance_id) || !hash(handoff.schema_hash))
        return false;
    std::set<std::string> unique;
    for (const auto& capability : handoff.capabilities)
        if (!known_capability(capability) || !unique.insert(capability).second) return false;
    return handoff.capabilities.size() <= 5;
}
} // namespace

content_handoff::~content_handoff() noexcept
{
    erase_bytes(token.data(), token.size());
}

launch_handoff::~launch_handoff() noexcept
{
    erase_bytes(secret.data(), secret.size());
}

bool read_launch_handoff(std::uintptr_t handle, launch_handoff& result, std::string& error)
{
    error.clear();
    std::array<std::uint8_t, 4> prefix{};
    if (!read_pipe(handle, prefix.data(), prefix.size()))
    {
        error = "Missing protected Live Play launch handoff";
        return false;
    }
    const std::uint32_t bytes = (std::uint32_t(prefix[0]) << 24) | (std::uint32_t(prefix[1]) << 16) |
                                (std::uint32_t(prefix[2]) << 8) | std::uint32_t(prefix[3]);
    if (!bytes || bytes > maximum_handoff_bytes)
    {
        error = "Invalid Live Play launch handoff size";
        return false;
    }
    std::string payload(bytes, '\0');
    if (!read_pipe(handle, payload.data(), bytes))
    {
        erase_bytes(payload.data(), payload.size());
        error = "Incomplete protected Live Play launch handoff";
        return false;
    }
    const auto parsed = json::parse(payload, nullptr, false);
    erase_bytes(payload.data(), payload.size());
    const bool has_content = parsed.is_object() && parsed.contains("content");
    if (parsed.is_discarded() ||
        !(has_content ? exact_keys(parsed, {"port", "secret", "projectGuid", "buildId", "targetProfile",
                                            "sessionId", "instanceId", "schemaHash", "capabilities", "content"})
                      : exact_keys(parsed, {"port", "secret", "projectGuid", "buildId", "targetProfile",
                                            "sessionId", "instanceId", "schemaHash", "capabilities"})))
    {
        error = "Invalid Live Play launch handoff schema";
        return false;
    }
    std::uint64_t port = 0;
    std::string secret;
    launch_handoff candidate;
    if (!safe_number(parsed, "port", port) || port == 0 || port > 65535 || !field(parsed, "secret", secret) ||
        secret.size() != 64 || !field(parsed, "projectGuid", candidate.project_guid) ||
        !field(parsed, "buildId", candidate.build_id) || !field(parsed, "targetProfile", candidate.target_profile) ||
        !field(parsed, "sessionId", candidate.session_id) || !field(parsed, "instanceId", candidate.instance_id) ||
        !field(parsed, "schemaHash", candidate.schema_hash) ||
        !parse_capabilities(parsed.at("capabilities"), candidate.capabilities))
    {
        erase_bytes(secret.data(), secret.size());
        error = "Invalid Live Play launch handoff fields";
        return false;
    }
    candidate.port = static_cast<std::uint16_t>(port);
    for (std::size_t i = 0; i < secret_bytes; ++i)
    {
        const auto digit = [](char c) -> int
        {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        const int high = digit(secret[2 * i]);
        const int low = digit(secret[2 * i + 1]);
        if (high < 0 || low < 0)
        {
            erase_bytes(secret.data(), secret.size());
            error = "Invalid Live Play handoff credential encoding";
            return false;
        }
        candidate.secret[i] = static_cast<std::uint8_t>((high << 4) | low);
    }
    erase_bytes(secret.data(), secret.size());
    if (has_content)
    {
        const auto& raw = parsed.at("content");
        content_handoff content;
        std::uint64_t generation{}, authoring_revision{};
        if (!exact_keys(raw, {"origin", "token", "sceneGuid", "sceneAssetGuid", "authoringSha256",
                              "authoringRevision", "descriptorSha256", "generation", "textureFamily"}) ||
            !field(raw, "origin", content.origin) || !field(raw, "token", content.token) ||
            !field(raw, "sceneGuid", content.scene_guid) ||
            !field(raw, "sceneAssetGuid", content.scene_asset_guid) ||
            !field(raw, "authoringSha256", content.authoring_sha256) ||
            !field(raw, "descriptorSha256", content.descriptor_sha256) ||
            !field(raw, "textureFamily", content.texture_family) ||
            !safe_number(raw, "generation", generation) || generation == 0 ||
            !safe_number(raw, "authoringRevision", authoring_revision) ||
            !guid(content.scene_guid) || !guid(content.scene_asset_guid) ||
            !hash(content.authoring_sha256) || !hash(content.descriptor_sha256) ||
            content.token.size() != 64 ||
            !std::all_of(content.token.begin(), content.token.end(),
                         [](unsigned char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'); }) ||
            content.texture_family.empty() || content.texture_family.size() > 32 ||
            !std::all_of(content.texture_family.begin(), content.texture_family.end(),
                         [](unsigned char ch) { return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'z') ||
                                                       ch == '_' || ch == '-'; }))
        {
            error = "Invalid private Live Play content pairing";
            return false;
        }
        const std::string_view prefix = "http://127.0.0.1:";
        if (!std::string_view(content.origin).starts_with(prefix))
        {
            error = "Live Play C9 pairing requires a local Development content listener";
            return false;
        }
        const auto port_text = std::string_view(content.origin).substr(prefix.size());
        std::uint32_t content_port{};
        auto [end, ec] = std::from_chars(port_text.data(), port_text.data() + port_text.size(), content_port);
        if (ec != std::errc{} || end != port_text.data() + port_text.size() || content_port == 0 ||
            content_port > 65535)
        {
            error = "Invalid Live Play C9 listener authority";
            return false;
        }
        content.generation = generation;
        content.authoring_revision = authoring_revision;
        candidate.content = std::move(content);
    }
    if (!valid_authority(candidate))
    {
        error = "Invalid Live Play session authority";
        return false;
    }
    result.port = candidate.port;
    result.secret = candidate.secret;
    result.project_guid = std::move(candidate.project_guid);
    result.build_id = std::move(candidate.build_id);
    result.target_profile = std::move(candidate.target_profile);
    result.session_id = std::move(candidate.session_id);
    result.instance_id = std::move(candidate.instance_id);
    result.schema_hash = std::move(candidate.schema_hash);
    result.capabilities = std::move(candidate.capabilities);
    result.content = std::move(candidate.content);
    return true;
}

client::~client() noexcept
{
    close();
}

bool client::connected() const noexcept
{
    return socket_ != -1;
}

void client::close() noexcept
{
    resume_outbound_sequence_ = std::max(resume_outbound_sequence_, outbound_sequence_);
    resume_inbound_sequence_ = std::max(resume_inbound_sequence_, inbound_sequence_);
    if (connected()) close_socket(native_socket(socket_));
    socket_ = -1;
    outbound_sequence_ = 0;
    inbound_sequence_ = 0;
    negotiated_capabilities_.clear();
    pending_baseline_request_.clear();
    pending_baseline_scene_.clear();
    pending_baseline_release_.clear();
    pending_baseline_generation_ = 0;
    project_guid_.clear();
    build_id_.clear();
    target_profile_.clear();
    session_id_.clear();
    instance_id_.clear();
#if defined(_WIN32)
    if (initialized_socket_api_) WSACleanup();
#endif
    initialized_socket_api_ = false;
}

bool client::connect(const launch_handoff& handoff, std::string& error)
{
    close();
    resume_outbound_sequence_ = 0;
    resume_inbound_sequence_ = 0;
    return connect_impl(handoff, error, false);
}

bool client::reconnect(const launch_handoff& handoff, std::string& error)
{
    close();
    return connect_impl(handoff, error, true);
}

bool client::connect_impl(const launch_handoff& handoff, std::string& error, bool resume)
{
    error.clear();
    outbound_sequence_ = resume ? resume_outbound_sequence_ : 0;
    inbound_sequence_ = resume ? resume_inbound_sequence_ : 0;
    if (!valid_authority(handoff))
    {
        error = "Invalid Live Play session authority";
        return false;
    }
#if defined(_WIN32)
    WSADATA information{};
    if (WSAStartup(MAKEWORD(2, 2), &information) != 0)
    {
        error = "Live Play network initialization failed";
        return false;
    }
    initialized_socket_api_ = true;
#endif
    const auto s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == invalid_socket)
    {
        error = "Could not create Live Play TCP socket";
        close();
        return false;
    }
    socket_ = static_cast<std::intptr_t>(s);
#if defined(SO_NOSIGPIPE)
    // macOS has SO_NOSIGPIPE rather than Linux/Android MSG_NOSIGNAL.
    const int no_sigpipe = 1;
    ::setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif
    sockaddr_in destination{};
    destination.sin_family = AF_INET;
    destination.sin_port = htons(handoff.port);
    destination.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(s, reinterpret_cast<const sockaddr*>(&destination), sizeof(destination)) != 0)
    {
        error = "Live Play editor endpoint is unavailable";
        close();
        return false;
    }
    project_guid_ = handoff.project_guid;
    build_id_ = handoff.build_id;
    target_profile_ = handoff.target_profile;
    session_id_ = handoff.session_id;
    instance_id_ = handoff.instance_id;

    if (!send_frame(s, handoff.secret.data(), handoff.secret.size()))
    {
        error = "Live Play credential transfer failed";
        close();
        return false;
    }
    const json hello = {{"schemaHash", handoff.schema_hash},
                        {"lastReceivedSequence", inbound_sequence_},
                        {"capabilities", handoff.capabilities}};
    if (!send_message("hello", hello.dump(), error))
    {
        close();
        return false;
    }

    std::string response;
    if (!read_frame(s, response, read_timeout_ms))
    {
        error = "Live Play editor handshake timed out";
        close();
        return false;
    }
    const auto result = json::parse(response, nullptr, false);
    std::uint64_t sequence = 0;
    std::string type;
    std::string sender;
    if (result.is_discarded() ||
        !exact_keys(result, {"protocol", "protocolVersion", "sender", "projectGuid", "buildId", "targetProfile",
                             "sessionId", "instanceId", "messageId", "sequence", "type", "payload"}) ||
        !result.at("protocol").is_string() || result.at("protocol") != "arc.live-play" ||
        !result.at("protocolVersion").is_number_integer() || result.at("protocolVersion") != protocol_version ||
        !field(result, "sender", sender) || sender != "editor" || !field(result, "type", type) ||
        type != "hello.accepted" || !safe_number(result, "sequence", sequence) ||
        (resume ? sequence <= inbound_sequence_ : sequence != 1) || !result.at("messageId").is_string() ||
        !token(result.at("messageId").get<std::string>()) || result.at("projectGuid") != project_guid_ ||
        result.at("buildId") != build_id_ || result.at("targetProfile") != target_profile_ ||
        result.at("sessionId") != session_id_ || result.at("instanceId") != instance_id_)
    {
        error = "Live Play editor handshake identity or protocol mismatch";
        close();
        return false;
    }
    const auto& payload = result.at("payload");
    std::uint64_t acknowledged = 0;
    if (!exact_keys(payload, {"capabilities", "lastReceivedSequence"}) ||
        !safe_number(payload, "lastReceivedSequence", acknowledged) || acknowledged != outbound_sequence_ ||
        !parse_capabilities(payload.at("capabilities"), negotiated_capabilities_))
    {
        error = "Invalid Live Play handshake capability response";
        close();
        return false;
    }
    for (const auto& capability : negotiated_capabilities_)
        if (std::find(handoff.capabilities.begin(), handoff.capabilities.end(), capability) ==
            handoff.capabilities.end())
        {
            error = "Live Play editor escalated capabilities";
            close();
            return false;
        }
    inbound_sequence_ = sequence;
    resume_inbound_sequence_ = sequence;
    return true;
}

bool client::send_message(std::string_view type, const std::string& payload_json, std::string& error)
{
    if (!connected())
    {
        error = "Live Play runtime is disconnected";
        return false;
    }
    const std::uint64_t sequence = outbound_sequence_ + 1;
    if (sequence > maximum_js_safe_integer)
    {
        error = "Live Play sequence exhausted";
        return false;
    }
    json payload = json::parse(payload_json, nullptr, false);
    if (payload.is_discarded())
    {
        error = "Invalid outbound Live Play payload";
        return false;
    }
    const json object = {{"protocol", "arc.live-play"},
                         {"protocolVersion", protocol_version},
                         {"sender", "runtime"},
                         {"projectGuid", project_guid_},
                         {"buildId", build_id_},
                         {"targetProfile", target_profile_},
                         {"sessionId", session_id_},
                         {"instanceId", instance_id_},
                         {"messageId", "runtime-" + std::to_string(sequence)},
                         {"sequence", sequence},
                         {"type", std::string(type)},
                         {"payload", std::move(payload)}};
    const auto serialized = object.dump();
    // A send may reach the editor even when the socket reports failure.
    // Reserve the sequence before IO so reconnect can never replay it.
    outbound_sequence_ = sequence;
    if (!send_frame(native_socket(socket_), serialized.data(), serialized.size()))
    {
        error = "Cannot send Live Play runtime frame";
        close();
        return false;
    }
    return true;
}

bool client::send_state(std::string_view state, std::uint64_t tick, std::uint64_t revision, std::string& error)
{
    if (state != "starting" && state != "running" && state != "paused" && state != "stopped" && state != "faulted")
    {
        error = "Unsupported Live Play state";
        return false;
    }
    if (tick > maximum_js_safe_integer || revision > maximum_js_safe_integer)
    {
        error = "Live Play revision exceeds wire precision";
        return false;
    }
    return send_message("session.state",
                        json{{"state", std::string(state)}, {"tick", tick}, {"revision", revision}}.dump(), error);
}

bool client::send_heartbeat(std::string& error)
{
    return send_message("heartbeat", json{{"lastReceivedSequence", inbound_sequence_}}.dump(), error);
}

bool client::send_ack(std::string_view request_id, std::uint64_t applied_revision, std::string& error)
{
    if (!token(request_id) || applied_revision > maximum_js_safe_integer)
    {
        error = "Invalid Live Play command ACK";
        return false;
    }
    return send_message(
        "command.ack",
        json{{"status", "applied"}, {"requestId", std::string(request_id)}, {"appliedRevision", applied_revision}}
            .dump(),
        error);
}

bool client::send_rejected_ack(std::string_view request_id, std::string_view code, std::string& error)
{
    if (!token(request_id) ||
        (code != "unknown_entity" && code != "stale_revision" && code != "permission_denied" &&
         code != "unsupported_operation" && code != "schema_mismatch" && code != "invalid_request" &&
         code != "restart_required" && code != "resync_required" && code != "busy"))
    {
        error = "Invalid Live Play command rejection";
        return false;
    }
    return send_message("command.ack",
                        json{{"status", "rejected"},
                             {"requestId", std::string(request_id)},
                             {"error", {{"code", std::string(code)}, {"message", "Runtime edit was not applied"}}}}
                            .dump(),
                        error);
}

bool client::send_hierarchy_snapshot(const hierarchy_snapshot& snapshot, std::string& error)
{
    if (std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(), "entity.inspect") ==
        negotiated_capabilities_.end())
    {
        error = "Runtime inspection is not negotiated";
        return false;
    }
    // 64 pages x 64 entities, with each JSON frame comfortably below 64 KiB.
    if (snapshot.entities.size() > 4096 || snapshot.world_revision > maximum_js_safe_integer)
    {
        error = "Runtime hierarchy snapshot exceeds protocol bounds";
        return false;
    }
    const auto encode_entity = [](const inspection_entity& entity) -> json
    {
        return entity.authored ? json{{"kind", "authored"}, {"entityGuid", entity.id}}
                               : json{{"kind", "runtime"}, {"runtimeEntityId", entity.id}};
    };
    const auto valid = [](const inspection_entity& entity)
    { return entity.authored ? guid(entity.id) : token(entity.id); };
    for (const auto& item : snapshot.entities)
    {
        if (item.name.size() > 256 || !valid(item.entity) || (item.parent && !valid(*item.parent)))
        {
            error = "Invalid runtime hierarchy entity";
            return false;
        }
    }

    constexpr std::size_t page_size = 64;
    const auto pages = std::max<std::size_t>(1, (snapshot.entities.size() + page_size - 1) / page_size);
    for (std::size_t page = 0; page < pages; ++page)
    {
        json entities = json::array();
        const auto end = std::min(snapshot.entities.size(), (page + 1) * page_size);
        for (auto i = page * page_size; i < end; ++i)
        {
            const auto& item = snapshot.entities[i];
            entities.push_back({{"entity", encode_entity(item.entity)},
                                {"parent", item.parent ? encode_entity(*item.parent) : json(nullptr)},
                                {"name", item.name}});
        }
        const json payload = {{"worldRevision", snapshot.world_revision},
                              {"entities", std::move(entities)},
                              {"complete", snapshot.complete},
                              {"pageIndex", page},
                              {"pageCount", pages}};
        if (!send_message("hierarchy.snapshot", payload.dump(-1, ' ', false, json::error_handler_t::replace), error))
            return false;
    }
    return true;
}

bool client::send_hierarchy_changed(const hierarchy_delta& delta, std::string& error)
{
    if (std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(), "entity.inspect") ==
            negotiated_capabilities_.end() ||
        delta.changes.size() > 64 || delta.world_revision <= delta.base_revision ||
        delta.world_revision > maximum_js_safe_integer)
    {
        error = "Invalid or unauthorized runtime hierarchy delta";
        return false;
    }
    const auto encode_entity = [](const inspection_entity& entity) -> json
    {
        return entity.authored ? json{{"kind", "authored"}, {"entityGuid", entity.id}}
                               : json{{"kind", "runtime"}, {"runtimeEntityId", entity.id}};
    };
    const auto valid = [](const inspection_entity& entity)
    { return entity.authored ? guid(entity.id) : token(entity.id); };
    json changes = json::array();
    for (const auto& change : delta.changes)
    {
        if (!valid(change.item.entity))
        {
            error = "Invalid runtime hierarchy delta entity";
            return false;
        }
        if (change.kind == hierarchy_change_kind::removed)
            changes.push_back({{"kind", "removed"}, {"entity", encode_entity(change.item.entity)}});
        else
        {
            if (change.item.name.size() > 256 || (change.item.parent && !valid(*change.item.parent)))
            {
                error = "Invalid runtime hierarchy upsert";
                return false;
            }
            changes.push_back({{"kind", "upsert"},
                               {"value",
                                {{"entity", encode_entity(change.item.entity)},
                                 {"parent", change.item.parent ? encode_entity(*change.item.parent) : json(nullptr)},
                                 {"name", change.item.name}}}});
        }
    }
    const json payload = {{"baseRevision", delta.base_revision},
                          {"worldRevision", delta.world_revision},
                          {"changes", std::move(changes)}};
    return send_message("hierarchy.changed", payload.dump(-1, ' ', false, json::error_handler_t::replace), error);
}

bool client::send_inspection_result(const component_inspection& result, std::string& error)
{
    if (std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(), "entity.inspect") ==
            negotiated_capabilities_.end() ||
        result.runtime_revision > maximum_js_safe_integer || result.properties.size() > 64)
    {
        error = "Runtime component inspection is not negotiated or exceeds bounds";
        return false;
    }
    const auto& target = result.request;
    const json entity = target.entity.authored ? json{{"kind", "authored"}, {"entityGuid", target.entity.id}}
                                               : json{{"kind", "runtime"}, {"runtimeEntityId", target.entity.id}};
    json properties = json::array();
    for (const auto& item : result.properties)
    {
        json value;
        if (item.kind == inspection_value_kind::numeric_array)
        {
            value = item.numbers;
            if (item.numbers.size() > 16) return false;
        }
        else if (item.kind == inspection_value_kind::boolean)
            value = item.flag;
        else
        {
            if (item.text.size() > 2048) return false;
            value = item.text;
        }
        properties.push_back({{"path", item.path}, {"value", std::move(value)}});
    }
    const json payload = {{"requestId", target.request_id},
                          {"entity", entity},
                          {"component", {{"typeId", target.component}, {"schemaVersion", target.schema_version}}},
                          {"runtimeRevision", result.runtime_revision},
                          {"properties", std::move(properties)}};
    return send_message("component.inspect.result", payload.dump(-1, ' ', false, json::error_handler_t::replace),
                        error);
}

bool client::send_content_generation(std::uint64_t generation, std::string_view status, std::string& error)
{
    if (status != "available" && status != "active" && status != "rejected")
    {
        error = "Invalid Live Play content generation state";
        return false;
    }
    if (generation > maximum_js_safe_integer)
    {
        error = "Live Play generation exceeds wire precision";
        return false;
    }
    if (std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(), "content.observe") ==
        negotiated_capabilities_.end())
    {
        error = "Content observation is not negotiated";
        return false;
    }
    return send_message("content.generation", json{{"generation", generation}, {"status", std::string(status)}}.dump(),
                        error);
}

bool client::send_baseline_status(std::string_view status, std::string_view reason, std::string& error)
{
    const bool content_negotiated = std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(),
                                              "content.observe") != negotiated_capabilities_.end();
    if (pending_baseline_request_.empty() || !content_negotiated)
    {
        error = "No negotiated Live Play cooked baseline offer is pending";
        return false;
    }
    if ((status == "active" && reason != "none") ||
        (status == "rejected" && reason != "unavailable" && reason != "incompatible" && reason != "corrupt" &&
         reason != "unsupported") ||
        (status != "active" && status != "rejected"))
    {
        error = "Invalid Live Play baseline result";
        return false;
    }
    const json payload = {{"requestId", pending_baseline_request_},
                          {"sceneGuid", pending_baseline_scene_},
                          {"cookedGeneration", pending_baseline_generation_},
                          {"cookedReleaseSha256", pending_baseline_release_},
                          {"status", std::string(status)},
                          {"reason", std::string(reason)}};
    if (!send_message("scene.baseline.status", payload.dump(), error)) return false;
    pending_baseline_request_.clear();
    pending_baseline_scene_.clear();
    pending_baseline_release_.clear();
    pending_baseline_generation_ = 0;
    return true;
}

bool client::receive_frame(std::string& result, std::uint32_t timeout_ms, std::string& error)
{
    if (!connected())
    {
        error = "Live Play runtime is disconnected";
        return false;
    }
    fd_set readable;
    FD_ZERO(&readable);
    const auto s = native_socket(socket_);
    FD_SET(s, &readable);
    timeval timeout{static_cast<long>(timeout_ms / 1000), static_cast<long>((timeout_ms % 1000) * 1000)};
#if defined(_WIN32)
    const int ready = ::select(0, &readable, nullptr, nullptr, &timeout);
#else
    const int ready = ::select(s + 1, &readable, nullptr, nullptr, &timeout);
#endif
    if (ready == 0) return false;
    if (ready < 0 || !read_frame(s, result, read_timeout_ms))
    {
        error = "Live Play editor connection closed or produced an invalid frame";
        close();
        return false;
    }
    return true;
}

std::optional<incoming_message> client::poll(std::uint32_t timeout_ms, std::string& error)
{
    error.clear();
    std::string text;
    if (!receive_frame(text, timeout_ms, error)) return std::nullopt;
    const auto parsed = json::parse(text, nullptr, false);
    std::string sender, type, message_id;
    std::uint64_t sequence = 0;
    if (parsed.is_discarded() ||
        !exact_keys(parsed, {"protocol", "protocolVersion", "sender", "projectGuid", "buildId", "targetProfile",
                             "sessionId", "instanceId", "messageId", "sequence", "type", "payload"}) ||
        parsed.at("protocol") != "arc.live-play" || parsed.at("protocolVersion") != protocol_version ||
        !field(parsed, "sender", sender) || sender != "editor" || !field(parsed, "type", type) ||
        !field(parsed, "messageId", message_id) || !token(message_id) || !safe_number(parsed, "sequence", sequence) ||
        sequence <= inbound_sequence_ || parsed.at("projectGuid") != project_guid_ ||
        parsed.at("buildId") != build_id_ || parsed.at("targetProfile") != target_profile_ ||
        parsed.at("sessionId") != session_id_ || parsed.at("instanceId") != instance_id_)
    {
        error = "Live Play inbound identity, direction, or sequence mismatch";
        close();
        return std::nullopt;
    }
    const auto& payload = parsed.at("payload");
    std::uint64_t seen = 0;
    std::optional<property_edit> edit;
    std::optional<inspection_request> inspection;
    // L1.3b1 recognizes cooked baseline metadata but does not touch ECS or
    // content IO. The game host must ask the verified C9 loader to activate it.
    if (type == "scene.baseline.offer")
    {
        std::string scene, authored_hash, release_hash;
        std::uint64_t authoring_revision = 0, generation = 0;
        if (!pending_baseline_request_.empty() ||
            std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(), "content.observe") ==
                negotiated_capabilities_.end() ||
            !exact_keys(payload, {"sceneGuid", "authoringRevision", "authoringSha256", "cookedGeneration",
                                  "cookedReleaseSha256"}) ||
            !field(payload, "sceneGuid", scene) || !guid(scene) ||
            !safe_number(payload, "authoringRevision", authoring_revision) ||
            !field(payload, "authoringSha256", authored_hash) || !hash(authored_hash) ||
            !safe_number(payload, "cookedGeneration", generation) ||
            !field(payload, "cookedReleaseSha256", release_hash) || !hash(release_hash))
        {
            error = "Invalid or unsolicited cooked scene baseline offer";
            close();
            return std::nullopt;
        }
        pending_baseline_request_ = message_id;
        pending_baseline_scene_ = std::move(scene);
        pending_baseline_release_ = std::move(release_hash);
        pending_baseline_generation_ = generation;
    }
    else if (type == "entity.property.set")
    {
        const bool allowed = std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(),
                                       "entity.edit") != negotiated_capabilities_.end();
        if (allowed) edit = decode_property_edit(message_id, payload.dump());
        if (!allowed || !edit)
        {
            error = "Invalid or unauthorized Live Play entity edit";
            close();
            return std::nullopt;
        }
    }
    else if (type == "component.inspect.request")
    {
        const bool allowed = std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(),
                                       "entity.inspect") != negotiated_capabilities_.end();
        if (allowed) inspection = decode_inspection_request(message_id, payload.dump());
        if (!allowed || !inspection)
        {
            error = "Invalid or unauthorized Live Play inspection request";
            close();
            return std::nullopt;
        }
    }
    else if (type == "session.resync.request")
    {
        std::string reason;
        const bool allowed = std::find(negotiated_capabilities_.begin(), negotiated_capabilities_.end(),
                                       "entity.inspect") != negotiated_capabilities_.end();
        if (!allowed || !exact_keys(payload, {"lastReceivedSequence", "reason"}) ||
            !safe_number(payload, "lastReceivedSequence", seen) || seen > outbound_sequence_ ||
            !field(payload, "reason", reason) || (reason != "reconnected" && reason != "gap" && reason != "overflow"))
        {
            error = "Invalid or unauthorized Live Play resynchronization request";
            close();
            return std::nullopt;
        }
    }
    else if (type != "heartbeat" || !exact_keys(payload, {"lastReceivedSequence"}) ||
             !safe_number(payload, "lastReceivedSequence", seen) || seen > outbound_sequence_)
    {
        error = "Live Play command is not yet supported by native Development bridge";
        close();
        return std::nullopt;
    }
    inbound_sequence_ = sequence;
    return incoming_message{std::move(type), std::move(message_id), sequence,
                            std::move(text), std::move(edit),       std::move(inspection)};
}

std::uint64_t client::last_received_sequence() const noexcept
{
    return inbound_sequence_;
}

} // namespace arc::network::live_play_dev
