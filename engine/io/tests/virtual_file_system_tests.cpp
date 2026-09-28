#include <arc/io/virtual_file_system.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <fstream>

namespace
{

arc::io::virtual_path path(std::string_view value)
{
    auto parsed = arc::io::virtual_path::parse(value);
    REQUIRE(parsed.succeeded());
    return std::move(parsed.value());
}

arc::io::file_buffer bytes(std::initializer_list<unsigned char> values)
{
    arc::io::file_buffer result;
    for (const auto value : values)
        result.push_back(static_cast<std::byte>(value));
    return result;
}

} // namespace

TEST_CASE("virtual paths normalize scheme authority and separators")
{
    auto parsed = arc::io::virtual_path::parse("PACKAGE://BASE//textures/./stone.arcimg");
    REQUIRE(parsed.succeeded());
    CHECK(parsed.value().string() == "package://base/textures/stone.arcimg");
    CHECK(parsed.value().scheme() == "package");
    CHECK(parsed.value().authority() == "base");
    CHECK(parsed.value().relative_path() == "textures/stone.arcimg");

    CHECK_FALSE(arc::io::virtual_path::parse("package://base/../secret").succeeded());
    CHECK_FALSE(arc::io::virtual_path::parse("package://base/a\\b").succeeded());
    CHECK_FALSE(arc::io::virtual_path::parse("C:/native/path").succeeded());
}

TEST_CASE("virtual filesystem resolves overlays tombstones and immutable generations")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::virtual_file_system vfs(jobs);
    auto base = std::make_shared<arc::io::memory_file_provider>(jobs);
    auto overlay = std::make_shared<arc::io::memory_file_provider>(jobs);
    const std::array base_files{arc::io::memory_provider_update{.relative_path = "mesh.bin", .bytes = bytes({1, 2, 3})},
                                arc::io::memory_provider_update{.relative_path = "gone.bin", .bytes = bytes({4})}};
    REQUIRE(base->publish(base_files).succeeded());
    REQUIRE(vfs.mount({.root = path("artifact://game"), .priority = 0, .provider = base}).succeeded());
    REQUIRE(vfs.mount({.root = path("artifact://game"), .priority = 100, .provider = overlay}).succeeded());

    auto original = vfs.resolve("artifact://game/mesh.bin");
    REQUIRE(original.succeeded());
    CHECK(vfs.read_all(original.value()).get().value() == bytes({1, 2, 3}));

    const std::array overlay_files{arc::io::memory_provider_update{.relative_path = "mesh.bin", .bytes = bytes({9, 8})},
                                   arc::io::memory_provider_update{.relative_path = "gone.bin", .tombstone = true}};
    REQUIRE(overlay->publish(overlay_files).succeeded());
    auto replacement = vfs.resolve("artifact://game/mesh.bin");
    REQUIRE(replacement.succeeded());
    CHECK(vfs.read_all(replacement.value()).get().value() == bytes({9, 8}));
    CHECK(vfs.read_all(original.value()).get().value() == bytes({1, 2, 3}));
    const std::array changed_base{arc::io::memory_provider_update{.relative_path = "mesh.bin", .bytes = bytes({7})}};
    REQUIRE(base->publish(changed_base).succeeded());
    const auto stale = vfs.read_all(original.value()).get();
    REQUIRE_FALSE(stale.succeeded());
    CHECK(stale.error().code == arc::io::file_error_code::stale_handle);
    auto removed = vfs.resolve("artifact://game/gone.bin");
    REQUIRE_FALSE(removed.succeeded());
    CHECK(removed.error().code == arc::io::file_error_code::tombstoned);
}

TEST_CASE("virtual filesystem change history is ordered bounded and caller-polled")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::io::virtual_file_system vfs(jobs, {.event_history_capacity = 2});
    auto provider = std::make_shared<arc::io::memory_file_provider>(jobs);
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = provider}).succeeded());
    std::size_t callback_count{};
    const auto subscription =
        vfs.subscribe([&](const arc::io::virtual_change_batch& batch) { callback_count += batch.events.size(); });
    const std::array updates{arc::io::memory_provider_update{.relative_path = "z.bin", .bytes = bytes({1})},
                             arc::io::memory_provider_update{.relative_path = "a.bin", .bytes = bytes({2})},
                             arc::io::memory_provider_update{.relative_path = "m.bin", .bytes = bytes({3})}};
    REQUIRE(provider->publish(updates).succeeded());
    CHECK(callback_count == 0);
    const auto batch = vfs.poll_changes();
    REQUIRE(batch.events.size() == 3);
    CHECK(batch.events[0].path.string() == "package://base/a.bin");
    CHECK(batch.events[1].path.string() == "package://base/m.bin");
    CHECK(batch.events[2].path.string() == "package://base/z.bin");
    CHECK(batch.history_overflow);
    CHECK(callback_count == 3);
    const auto history = vfs.events_since(0);
    CHECK(history.history_overflow);
    CHECK(history.events.size() == 2);
    vfs.unsubscribe(subscription);
}

TEST_CASE("virtual filesystem rejects ambiguous priorities and unmounts exact providers")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false});
    arc::io::virtual_file_system vfs(jobs);
    auto first = std::make_shared<arc::io::memory_file_provider>(jobs);
    auto second = std::make_shared<arc::io::memory_file_provider>(jobs);
    const std::array update{arc::io::memory_provider_update{.relative_path = "data.bin", .bytes = bytes({1})}};
    REQUIRE(first->publish(update).succeeded());
    const auto mounted = vfs.mount({.root = path("package://base"), .priority = 5, .provider = first});
    REQUIRE(mounted.succeeded());
    CHECK_FALSE(vfs.mount({.root = path("package://base"), .priority = 5, .provider = second}).succeeded());
    REQUIRE(vfs.resolve("package://base/data.bin").succeeded());
    REQUIRE(vfs.unmount(mounted.value()).succeeded());
    CHECK_FALSE(vfs.resolve("package://base/data.bin").succeeded());
}

TEST_CASE("filesystem provider enforces its root and reports changes")
{
    arc::jobs::job_system jobs(
        {.worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false});
    arc::io::async_file_service files(jobs);
    const auto root = std::filesystem::temp_directory_path() / "arc_vfs_filesystem_provider";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    {
        std::ofstream output(root / "data.bin", std::ios::binary);
        output << "abcd";
    }
    auto provider = std::make_shared<arc::io::filesystem_file_provider>(
        files, root, arc::io::filesystem_provider_config{.debounce = std::chrono::milliseconds{0}});
    arc::io::virtual_file_system vfs(jobs);
    REQUIRE(vfs.mount({.root = path("package://base"), .provider = provider}).succeeded());
    auto resolved = vfs.resolve("package://base/data.bin");
    REQUIRE(resolved.succeeded());
    auto range = vfs.read_range(resolved.value(), 1, 2).get();
    REQUIRE(range.succeeded());
    CHECK(range.value() == bytes({'b', 'c'}));
    CHECK_FALSE(vfs.resolve("package://base/../outside.bin").succeeded());

    CHECK(vfs.poll_changes().events.empty());
    {
        std::ofstream output(root / "added.bin", std::ios::binary);
        output << "new";
    }
    const auto changes = vfs.poll_changes();
    REQUIRE(changes.events.size() == 1);
    CHECK(changes.events.front().kind == arc::io::virtual_change_kind::added);
    CHECK(changes.events.front().path.string() == "package://base/added.bin");
    std::filesystem::remove_all(root);
}
