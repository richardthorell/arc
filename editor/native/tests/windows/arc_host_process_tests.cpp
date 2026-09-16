#include "../editor_test_support.h"

#include <arc/editor/arc_host.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>

using arc::editor::tests::parse_entity_from_response;

TEST_CASE("arc host process speaks newline delimited json over stdio")
{
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;

    HANDLE child_stdin_read = nullptr;
    HANDLE child_stdin_write = nullptr;
    HANDLE child_stdout_read = nullptr;
    HANDLE child_stdout_write = nullptr;
    REQUIRE(CreatePipe(&child_stdin_read, &child_stdin_write, &security, 0));
    REQUIRE(CreatePipe(&child_stdout_read, &child_stdout_write, &security, 0));
    REQUIRE(SetHandleInformation(child_stdin_write, HANDLE_FLAG_INHERIT, 0));
    REQUIRE(SetHandleInformation(child_stdout_read, HANDLE_FLAG_INHERIT, 0));

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_stdin_read;
    startup.hStdOutput = child_stdout_write;
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    PROCESS_INFORMATION process{};
    std::string command_line = "\"" ARC_HOST_PROCESS_PATH "\"";
    REQUIRE(
        CreateProcessA(nullptr, command_line.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startup, &process));

    CloseHandle(child_stdin_read);
    CloseHandle(child_stdout_write);

    const auto read_line = [&]()
    {
        std::string line;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline)
        {
            DWORD available = 0;
            REQUIRE(PeekNamedPipe(child_stdout_read, nullptr, 0, nullptr, &available, nullptr));
            if (available == 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            char ch{};
            DWORD read = 0;
            REQUIRE(ReadFile(child_stdout_read, &ch, 1, &read, nullptr));
            if (read == 0) continue;
            if (ch == '\n') return line;
            if (ch != '\r') line.push_back(ch);
        }
        return line;
    };

    std::vector<std::string> observed_events;
    const auto request = [&](std::uint64_t request_id, const std::string& json)
    {
        const std::string line = json + '\n';
        DWORD written = 0;
        REQUIRE(WriteFile(child_stdin_write, line.data(), static_cast<DWORD>(line.size()), &written, nullptr));
        REQUIRE(written == line.size());

        for (;;)
        {
            auto response = read_line();
            REQUIRE_FALSE(response.empty());
            if (response.find("\"kind\":\"response\"") != std::string::npos &&
                response.find("\"requestId\":" + std::to_string(request_id)) != std::string::npos)
            {
                REQUIRE(response.find("\"succeeded\":true") != std::string::npos);
                return response;
            }
            observed_events.push_back(std::move(response));
        }
    };

    request(1, arc::editor::to_json(arc::editor::host_command_envelope{
                   .request_id = 1,
                   .payload = arc::editor::host_open_project_command{.name = "Process Test",
                                                                     .root = std::filesystem::temp_directory_path()}}));

    const auto create_response = request(
        2,
        arc::editor::to_json(arc::editor::host_command_envelope{
            .request_id = 2,
            .payload = arc::editor::host_create_entity_command{.kind = arc::editor::host_create_entity_kind::cube}}));
    const auto created_entity = parse_entity_from_response(create_response);
    REQUIRE(created_entity.valid());
    REQUIRE(create_response.find("\"guid\":") != std::string::npos);

    const auto hierarchy_response =
        request(3, arc::editor::to_json(arc::editor::host_query_envelope{
                       .request_id = 3, .payload = arc::editor::host_scene_hierarchy_query{}}));
    REQUIRE(hierarchy_response.find("\"entities\"") != std::string::npos);

    request(4,
            arc::editor::to_json(arc::editor::host_command_envelope{
                .request_id = 4,
                .payload = arc::editor::host_rename_entity_command{.entity = created_entity, .name = "Process Cube"}}));

    const auto renamed_hierarchy =
        request(5, arc::editor::to_json(arc::editor::host_query_envelope{
                       .request_id = 5, .payload = arc::editor::host_scene_hierarchy_query{}}));
    REQUIRE(renamed_hierarchy.find("Process Cube") != std::string::npos);

    request(6, arc::editor::to_json(arc::editor::host_command_envelope{
                   .request_id = 6, .payload = arc::editor::host_close_project_command{}}));

    CloseHandle(child_stdin_write);
    WaitForSingleObject(process.hProcess, 5000);
    CloseHandle(child_stdout_read);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);

    REQUIRE(std::any_of(observed_events.begin(), observed_events.end(), [](const auto& event)
                        { return event.find("\"type\":\"entity.created\"") != std::string::npos; }));
}
