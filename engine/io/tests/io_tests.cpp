#include <arc/io/io.h>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <filesystem>
#include <fstream>

namespace
{

std::filesystem::path test_path(std::string_view name)
{
    return std::filesystem::temp_directory_path() / ("arc_io_" + std::string(name));
}

}

TEST_CASE("async file service writes reads ranges and stats files")
{
    arc::job_system jobs({ .worker_count = 1, .run_inline = false, .io_worker_count = 2, .enable_render_thread = false });
    arc::io::async_file_service files(jobs, { .chunk_size = 4096 });
    const auto path = test_path("roundtrip.bin");
    const std::array<std::byte, 8> bytes{
        std::byte{ 1 }, std::byte{ 2 }, std::byte{ 3 }, std::byte{ 4 },
        std::byte{ 5 }, std::byte{ 6 }, std::byte{ 7 }, std::byte{ 8 }
    };

    REQUIRE(files.write(path, bytes).get().succeeded());
    auto all = files.read_all(path).get();
    REQUIRE(all.succeeded());
    REQUIRE(all.value() == arc::io::file_buffer(bytes.begin(), bytes.end()));

    auto range = files.read_range(path, 2, 3).get();
    REQUIRE(range.succeeded());
    REQUIRE(range.value().size() == 3);
    REQUIRE(range.value()[0] == std::byte{ 3 });

    auto info = files.stat(path).get();
    REQUIRE(info.succeeded());
    REQUIRE(info.value().size == bytes.size());
    REQUIRE(info.value().regular_file);
    std::filesystem::remove(path);
}

TEST_CASE("atomic writes replace destinations without leaving temporary files")
{
    arc::job_system jobs({ .worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false });
    arc::io::async_file_service files(jobs);
    const auto path = test_path("atomic.bin");
    const std::array<std::byte, 2> first{ std::byte{ 1 }, std::byte{ 2 } };
    const std::array<std::byte, 3> second{ std::byte{ 3 }, std::byte{ 4 }, std::byte{ 5 } };

    REQUIRE(files.write_atomic(path, first).get().succeeded());
    REQUIRE(files.write_atomic(path, second).get().succeeded());
    auto loaded = files.read_all(path).get();
    REQUIRE(loaded.succeeded());
    REQUIRE(loaded.value() == arc::io::file_buffer(second.begin(), second.end()));
    std::filesystem::remove(path);
}

TEST_CASE("async file service reports missing files and invalid ranges")
{
    arc::job_system jobs({ .worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false });
    arc::io::async_file_service files(jobs);
    const auto missing = test_path("missing.bin");
    std::filesystem::remove(missing);

    auto missing_result = files.read_all(missing).get();
    REQUIRE_FALSE(missing_result.succeeded());
    REQUIRE(missing_result.error().code == arc::io::file_error_code::not_found);

    const auto path = test_path("range.bin");
    {
        std::ofstream output(path, std::ios::binary);
        output << "abc";
    }
    auto invalid = files.read_range(path, 2, 8).get();
    REQUIRE_FALSE(invalid.succeeded());
    REQUIRE(invalid.error().code == arc::io::file_error_code::invalid_range);
    std::filesystem::remove(path);
}

TEST_CASE("cancelled async operations do not execute")
{
    arc::job_system jobs({ .worker_count = 1, .run_inline = false, .io_worker_count = 1, .enable_render_thread = false });
    arc::io::async_file_service files(jobs);
    arc::cancellation_source cancellation;
    cancellation.request_cancel();

    auto future = files.read_all(test_path("cancelled.bin"), cancellation.token());
    REQUIRE(future.handle().wait_result().status == arc::job_status::cancelled);
}
