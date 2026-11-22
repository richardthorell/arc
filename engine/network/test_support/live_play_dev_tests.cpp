#include <arc/network/live_play_dev_client.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace
{
using nlohmann::json;
using namespace arc::network::live_play_dev;
#if defined(_WIN32)
using native_socket = SOCKET;
constexpr native_socket invalid_socket = INVALID_SOCKET;
void close_socket(native_socket s)
{
    if (s != invalid_socket) closesocket(s);
}
#else
using native_socket = int;
constexpr native_socket invalid_socket = -1;
void close_socket(native_socket s)
{
    if (s != invalid_socket) ::close(s);
}
#endif

void check(bool condition, const char* why)
{
    if (!condition) throw std::runtime_error(why);
}

void send_bytes(native_socket s, const void* data, std::size_t count)
{
    const auto* pointer = static_cast<const char*>(data);
    while (count)
    {
        const auto sent = ::send(s, pointer, static_cast<int>(count), 0);
        check(sent > 0, "fixture server send failed");
        count -= static_cast<std::size_t>(sent);
        pointer += sent;
    }
}

void read_bytes(native_socket s, void* data, std::size_t count)
{
    auto* pointer = static_cast<char*>(data);
    while (count)
    {
        const auto received = ::recv(s, pointer, static_cast<int>(count), 0);
        check(received > 0, "fixture server receive failed");
        count -= static_cast<std::size_t>(received);
        pointer += received;
    }
}

void send_packet(native_socket s, const std::string& value)
{
    const std::size_t count = value.size();
    check(count <= maximum_frame_bytes, "test frame too large");
    const std::array<unsigned char, 4> prefix{
        static_cast<unsigned char>((count >> 24) & 255), static_cast<unsigned char>((count >> 16) & 255),
        static_cast<unsigned char>((count >> 8) & 255), static_cast<unsigned char>(count & 255)};
    send_bytes(s, prefix.data(), prefix.size());
    send_bytes(s, value.data(), count);
}

std::string read_packet(native_socket s)
{
    std::array<unsigned char, 4> prefix{};
    read_bytes(s, prefix.data(), prefix.size());
    const std::uint32_t count = (std::uint32_t(prefix[0]) << 24) | (std::uint32_t(prefix[1]) << 16) |
                                (std::uint32_t(prefix[2]) << 8) | std::uint32_t(prefix[3]);
    check(count > 0 && count <= maximum_frame_bytes, "fixture server received invalid size");
    std::string payload(count, '\0');
    read_bytes(s, payload.data(), payload.size());
    return payload;
}

std::string launch_record(std::uint16_t port)
{
    return json{{"port", port},
                {"secret", std::string(64, '1')},
                {"projectGuid", "11111111-1111-1111-1111-111111111111"},
                {"buildId", "dev-build-42"},
                {"targetProfile", "windows-x64-vulkan"},
                {"sessionId", "session-1"},
                {"instanceId", "runtime-1"},
                {"schemaHash", std::string(64, 'a')},
                {"capabilities", {"session.observe", "content.observe"}}}
        .dump();
}

launch_handoff inherited_handoff(std::uint16_t port)
{
    const auto record = launch_record(port);
    const std::size_t count = record.size();
    const std::array<unsigned char, 4> prefix{
        static_cast<unsigned char>((count >> 24) & 255), static_cast<unsigned char>((count >> 16) & 255),
        static_cast<unsigned char>((count >> 8) & 255), static_cast<unsigned char>(count & 255)};
    launch_handoff result;
    std::string error;
#if defined(_WIN32)
    HANDLE readable = nullptr, writable = nullptr;
    check(CreatePipe(&readable, &writable, nullptr, 0) == TRUE, "Could not create protected handoff pipe");
    DWORD sent = 0;
    check(WriteFile(writable, prefix.data(), static_cast<DWORD>(prefix.size()), &sent, nullptr) &&
              sent == prefix.size(),
          "Could not write handoff prefix");
    check(WriteFile(writable, record.data(), static_cast<DWORD>(record.size()), &sent, nullptr) &&
              sent == record.size(),
          "Could not write handoff record");
    CloseHandle(writable);
    const bool okay = read_launch_handoff(reinterpret_cast<std::uintptr_t>(readable), result, error);
    CloseHandle(readable);
#else
    int pipes[2]{-1, -1};
    check(::pipe(pipes) == 0, "Could not create protected handoff pipe");
    check(::write(pipes[1], prefix.data(), prefix.size()) == static_cast<ssize_t>(prefix.size()),
          "Could not write handoff prefix");
    check(::write(pipes[1], record.data(), record.size()) == static_cast<ssize_t>(record.size()),
          "Could not write handoff record");
    ::close(pipes[1]);
    const bool okay = read_launch_handoff(static_cast<std::uintptr_t>(pipes[0]), result, error);
    ::close(pipes[0]);
#endif
    check(okay, "Native protected inherited-pipe handoff was rejected");
    check(result.port == port && result.secret[0] == 0x11, "Handoff altered port or credential");
    check(result.schema_hash == std::string(64, 'a'), "Handoff altered schema hash");
    return result;
}

class fixture_server
{
public:
    fixture_server()
    {
#if defined(_WIN32)
        WSADATA information{};
        check(WSAStartup(MAKEWORD(2, 2), &information) == 0, "Fixture Winsock init failed");
#endif
        listener_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        check(listener_ != invalid_socket, "Fixture socket failed");
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = 0;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        check(::bind(listener_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "Fixture bind failed");
        check(::listen(listener_, 1) == 0, "Fixture listen failed");
#if defined(_WIN32)
        int length = sizeof(addr);
#else
        socklen_t length = sizeof(addr);
#endif
        check(::getsockname(listener_, reinterpret_cast<sockaddr*>(&addr), &length) == 0, "Fixture port missing");
        port_ = ntohs(addr.sin_port);
    }

    ~fixture_server()
    {
        close_socket(listener_);
#if defined(_WIN32)
        WSACleanup();
#endif
    }

    std::uint16_t port() const
    {
        return port_;
    }

    void serve(bool wrong_identity = false)
    {
        const auto peer = ::accept(listener_, nullptr, nullptr);
        check(peer != invalid_socket, "Fixture accept failed");
        try
        {
            const auto credential = read_packet(peer);
            check(credential.size() == secret_bytes && static_cast<unsigned char>(credential[0]) == 0x11,
                  "Incorrect raw credential framing");
            const auto hello = json::parse(read_packet(peer));
            std::ifstream fixture_file(ARC_LIVE_PLAY_WIRE_VECTOR_PATH);
            check(static_cast<bool>(fixture_file), "Cannot open cross-language golden protocol fixtures");
            const auto fixtures = json::parse(fixture_file);
            check(hello == fixtures.at("hello"), "Native hello does not match TypeScript v1 golden frame");
            check(hello.at("protocol") == "arc.live-play" && hello.at("protocolVersion") == 1,
                  "Native hello protocol mismatch");
            check(hello.at("sender") == "runtime" && hello.at("type") == "hello", "Native hello wrong direction");
            check(hello.at("projectGuid") == "11111111-1111-1111-1111-111111111111", "Native hello identity mismatch");
            check(hello.at("payload").at("schemaHash") == std::string(64, 'a'), "Native hello schema hash mismatch");
            check(hello.at("payload").at("capabilities") == json::array({"session.observe", "content.observe"}),
                  "Native hello capability mismatch");
            json accepted = hello;
            accepted["sender"] = "editor";
            accepted["sequence"] = 1;
            accepted["messageId"] = "editor-welcome";
            accepted["type"] = "hello.accepted";
            accepted["payload"] = {{"lastReceivedSequence", 1},
                                   {"capabilities", {"session.observe", "content.observe"}}};
            if (wrong_identity) accepted["sessionId"] = "incorrect-session";
            send_packet(peer, accepted.dump());
            if (!wrong_identity)
            {
                const auto state = json::parse(read_packet(peer));
                check(state.at("type") == "session.state", "Native state frame not received");
                check(state.at("payload").at("state") == "running" && state.at("payload").at("tick") == 7,
                      "Native state data incorrect");
                json heartbeat = accepted;
                heartbeat["type"] = "heartbeat";
                heartbeat["messageId"] = "editor-heartbeat-2";
                heartbeat["sequence"] = 2;
                heartbeat["payload"] = {{"lastReceivedSequence", 2}};
                check(heartbeat == fixtures.at("editorHeartbeat"),
                      "Mock editor heartbeat does not match TypeScript v1 golden frame");
                send_packet(peer, heartbeat.dump());
                const auto response = json::parse(read_packet(peer));
                check(response.at("type") == "heartbeat", "Native heartbeat response not received");
                const auto offered = fixtures.at("editorBaselineOffer");
                check(offered.at("sender") == "editor" && offered.at("sequence") == 3,
                      "Cross-language cooked baseline fixture is malformed");
                send_packet(peer, offered.dump());
                const auto baseline = json::parse(read_packet(peer));
                check(baseline.at("type") == "scene.baseline.status", "Native baseline result was not emitted");
                check(baseline.at("payload").at("requestId") == offered.at("messageId"),
                      "Native baseline ACK was not bound to the offer");
                check(baseline.at("payload").at("cookedReleaseSha256") ==
                          offered.at("payload").at("cookedReleaseSha256"),
                      "Native baseline report referenced a different cooked release");
                check(baseline.at("payload").at("status") == "rejected" &&
                          baseline.at("payload").at("reason") == "unavailable",
                      "Native baseline report must not claim content activated by the transport");
            }
        }
        catch (...)
        {
            close_socket(peer);
            throw;
        }
        close_socket(peer);
    }

private:
    native_socket listener_{invalid_socket};
    std::uint16_t port_{};
};

void test_successful_connection()
{
    fixture_server fixture;
    auto handoff = inherited_handoff(fixture.port());
    std::exception_ptr server_error;
    std::jthread server(
        [&]
        {
            try
            {
                fixture.serve();
            }
            catch (...)
            {
                server_error = std::current_exception();
            }
        });

    std::string error;
    client native;
    check(native.connect(handoff, error), "Native client failed to connect and negotiate Live Play");
    check(native.connected(), "Native client did not report connected");
    check(native.send_state("running", 7, 1, error), "Native runtime state did not send");
    const auto frame = native.poll(2000, error);
    check(frame.has_value() && frame->type == "heartbeat" && frame->sequence == 2,
          "Native client did not parse validated editor heartbeat");
    check(native.last_received_sequence() == 2, "Native inbound sequence was not advanced");
    check(native.send_heartbeat(error), "Native heartbeat did not send");
    check(!native.send_baseline_status("active", "none", error),
          "Native bridge acknowledged a baseline before receiving an offer");
    const auto offer = native.poll(2000, error);
    check(offer.has_value() && offer->type == "scene.baseline.offer" && offer->sequence == 3,
          "Native bridge did not validate the cooked baseline offer");
    check(native.last_received_sequence() == 3, "Native baseline sequence was not advanced");
    // Only a separate C9 scene loader may report 'active'. This transport
    // fixture deliberately reports no cooked release has been materialized.
    check(native.send_baseline_status("rejected", "unavailable", error),
          "Native cooked baseline rejection ACK was not transmitted");
    const bool duplicate_ack = native.send_baseline_status("rejected", "unavailable", error);
    check(!duplicate_ack, "Native bridge sent duplicate baseline ACK");
    native.close();
    server.join();
    if (server_error) std::rethrow_exception(server_error);
}

void test_property_edit_decoding_and_mailbox()
{
    using arc::network::live_play_dev::decode_property_edit;
    using arc::network::live_play_dev::edit_mailbox;
    const json valid = {
        {"entity", {{"kind", "authored"}, {"entityGuid", "33333333-3333-3333-3333-333333333333"}}},
        {"component", {{"typeId", "arc.transform"}, {"schemaVersion", 1}}},
        {"property", "position"},
        {"value", {2.0, 1.5, 0.0}},
        {"authoringRevision", 12},
        {"expectedRuntimeRevision", 8},
        {"phase", "commit"}};
    auto edit = decode_property_edit("editor-edit-42", valid.dump());
    check(edit.has_value(), "Valid native authored transform was rejected");
    check(edit->entity_guid == "33333333-3333-3333-3333-333333333333" &&
              edit->value_count == 3 && edit->values[0] == 2.0f &&
              edit->authoring_revision == 12 && edit->expected_runtime_revision == 8,
          "Decoded native transform differs from editor protocol");
    check(!decode_property_edit("bad request", valid.dump()), "Invalid message ID was accepted");
    auto mutated = valid;
    mutated["entity"]["kind"] = "runtime";
    check(!decode_property_edit("edit-1", mutated.dump()), "Runtime-only ECS identity was accepted");
    mutated = valid;
    mutated["entity"]["entityGuid"] = "non-guid";
    check(!decode_property_edit("edit-1", mutated.dump()), "Invalid stable GUID was accepted");
    mutated = valid;
    mutated["value"] = json::array({2.0, "invalid", 0.0});
    check(!decode_property_edit("edit-1", mutated.dump()), "Non-numeric transform was accepted");
    mutated = valid;
    mutated["value"] = json::array({2.0, 1.0, 0.0, 0.0, 0.0});
    check(!decode_property_edit("edit-1", mutated.dump()), "Oversized transform was accepted");
    mutated = valid;
    mutated["extra"] = 1;
    check(!decode_property_edit("edit-1", mutated.dump()), "Unexpected property field was accepted");
    mutated = valid;
    mutated["authoringRevision"] = -1;
    check(!decode_property_edit("edit-1", mutated.dump()), "Negative authoring revision was accepted");
    mutated = valid;
    mutated["expectedRuntimeRevision"] = 9007199254740992ULL;
    check(!decode_property_edit("edit-1", mutated.dump()), "Unsafe JavaScript revision was accepted");
    mutated = valid;
    mutated["phase"] = "publish";
    check(!decode_property_edit("edit-1", mutated.dump()), "Unknown edit phase was accepted");

    edit_mailbox mailbox;
    for (int i = 0; i < 256; ++i)
    {
        auto candidate = *edit;
        candidate.request_id = "edit-" + std::to_string(i);
        check(mailbox.enqueue(std::move(candidate)), "Mailbox rejected in-bounds edit");
    }
    check(!mailbox.enqueue(*edit), "Mailbox accepted unbounded producer backlog");
    const auto received = mailbox.take_edits();
    check(received.size() == 256 && received.front().request_id == "edit-0" &&
              received.back().request_id == "edit-255",
          "Mailbox dropped or reordered authored edits");
    check(mailbox.take_edits().empty(), "Mailbox replayed a drained edit");
    check(mailbox.report({.request_id = "edit-0", .applied = true, .applied_revision = 42}),
          "Runtime result could not be handed back to the socket owner");
    const auto outcomes = mailbox.take_results();
    check(outcomes.size() == 1 && outcomes.front().applied && outcomes.front().applied_revision == 42,
          "Mailbox lost applied revision ACK");
    mailbox.clear();
    check(mailbox.take_results().empty(), "Stop leaked old runtime results");
}

void test_invalid_editor_identity()
{
    fixture_server fixture;
    auto handoff = inherited_handoff(fixture.port());
    std::exception_ptr server_error;
    std::jthread server(
        [&]
        {
            try
            {
                fixture.serve(true);
            }
            catch (...)
            {
                server_error = std::current_exception();
            }
        });

    std::string error;
    client native;
    check(!native.connect(handoff, error), "Native client accepted an editor from a different session");
    check(!native.connected(), "Native client kept mismatched editor socket alive");
    server.join();
    if (server_error) std::rethrow_exception(server_error);
}

} // namespace

int main()
{
    try
    {
        test_successful_connection();
        test_invalid_editor_identity();
        test_property_edit_decoding_and_mailbox();
    }
    catch (const std::exception& error)
    {
        std::cerr << "Native Live Play Development transport test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
