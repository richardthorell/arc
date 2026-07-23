#include <arc/io/io.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <system_error>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#endif

namespace arc::io
{
namespace
{

file_error make_error(file_error_code code, const std::filesystem::path& path, std::string message)
{
    return { .code = code, .path = path, .message = std::move(message) };
}

file_error open_error(const std::filesystem::path& path)
{
    std::error_code error;
    if (!std::filesystem::exists(path, error))
        return make_error(file_error_code::not_found, path, "File does not exist");
    return make_error(file_error_code::permission_denied, path, "File could not be opened");
}

bool replace_file(const std::filesystem::path& temporary, const std::filesystem::path& destination) noexcept
{
#if defined(_WIN32)
    return MoveFileExW(
        temporary.c_str(),
        destination.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
#else
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    return !error;
#endif
}

std::filesystem::path temporary_path_for(const std::filesystem::path& destination)
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return destination.parent_path() /
        (destination.filename().string() + ".arc-tmp-" + std::to_string(stamp));
}

file_result<file_buffer> read_range_sync(
    const std::filesystem::path& path,
    std::uint64_t offset,
    std::optional<std::size_t> requested_bytes,
    std::size_t chunk_size,
    const cancellation_token& cancellation)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream)
        return file_result<file_buffer>::failure(open_error(path));

    const auto end = stream.tellg();
    if (end < 0)
        return file_result<file_buffer>::failure(make_error(file_error_code::read_failed, path, "File size could not be read"));
    const auto file_size = static_cast<std::uint64_t>(end);
    if (offset > file_size)
        return file_result<file_buffer>::failure(make_error(file_error_code::invalid_range, path, "Read offset is beyond the end of the file"));

    const auto available = file_size - offset;
    const auto desired = requested_bytes.has_value()
        ? static_cast<std::uint64_t>(*requested_bytes)
        : available;
    if (desired > available)
        return file_result<file_buffer>::failure(make_error(file_error_code::invalid_range, path, "Read range extends beyond the end of the file"));
    if (desired > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        return file_result<file_buffer>::failure(make_error(file_error_code::invalid_range, path, "File is too large for this process"));

    file_buffer result(static_cast<std::size_t>(desired));
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::size_t completed{};
    while (completed < result.size())
    {
        if (cancellation.stop_requested())
            return file_result<file_buffer>::failure(make_error(file_error_code::cancelled, path, "Read was cancelled"));
        const auto count = std::min(chunk_size, result.size() - completed);
        stream.read(reinterpret_cast<char*>(result.data() + completed), static_cast<std::streamsize>(count));
        if (stream.gcount() != static_cast<std::streamsize>(count))
            return file_result<file_buffer>::failure(make_error(file_error_code::read_failed, path, "File read ended unexpectedly"));
        completed += count;
    }
    return file_result<file_buffer>::success(std::move(result));
}

file_result<void> write_sync(
    const std::filesystem::path& path,
    const file_buffer& bytes,
    std::size_t chunk_size,
    const cancellation_token& cancellation)
{
    std::error_code directory_error;
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path(), directory_error);
    if (directory_error)
        return file_result<void>::failure(make_error(file_error_code::permission_denied, path, "Destination directory could not be created"));

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream)
        return file_result<void>::failure(make_error(file_error_code::permission_denied, path, "File could not be opened for writing"));
    std::size_t completed{};
    while (completed < bytes.size())
    {
        if (cancellation.stop_requested())
        {
            stream.close();
            return file_result<void>::failure(make_error(file_error_code::cancelled, path, "Write was cancelled"));
        }
        const auto count = std::min(chunk_size, bytes.size() - completed);
        stream.write(reinterpret_cast<const char*>(bytes.data() + completed), static_cast<std::streamsize>(count));
        if (!stream)
            return file_result<void>::failure(make_error(file_error_code::write_failed, path, "File write failed"));
        completed += count;
    }
    stream.flush();
    if (!stream)
        return file_result<void>::failure(make_error(file_error_code::write_failed, path, "File flush failed"));
    return file_result<void>::success();
}

}

async_file_service::async_file_service(job_system& jobs, async_file_config config)
    : jobs_(&jobs)
    , config_(config)
{
    config_.chunk_size = std::max<std::size_t>(config_.chunk_size, 4096);
}

job_future<file_result<file_buffer>> async_file_service::read_all(
    std::filesystem::path path,
    cancellation_token cancellation)
{
    return jobs_->submit_future({
        .name = "io.read_all",
        .priority = job_priority::normal,
        .affinity = job_affinity::io_thread,
        .cancellation = cancellation
    }, [path = std::move(path), cancellation, chunk = config_.chunk_size] {
        return read_range_sync(path, 0, std::nullopt, chunk, cancellation);
    });
}

job_future<file_result<file_buffer>> async_file_service::read_range(
    std::filesystem::path path,
    std::uint64_t offset,
    std::size_t bytes,
    cancellation_token cancellation)
{
    return jobs_->submit_future({
        .name = "io.read_range",
        .priority = job_priority::normal,
        .affinity = job_affinity::io_thread,
        .cancellation = cancellation
    }, [path = std::move(path), offset, bytes, cancellation, chunk = config_.chunk_size] {
        return read_range_sync(path, offset, bytes, chunk, cancellation);
    });
}

job_future<file_result<void>> async_file_service::write(
    std::filesystem::path path,
    std::span<const std::byte> bytes,
    cancellation_token cancellation)
{
    file_buffer owned(bytes.begin(), bytes.end());
    return jobs_->submit_future({
        .name = "io.write",
        .priority = job_priority::normal,
        .affinity = job_affinity::io_thread,
        .cancellation = cancellation
    }, [path = std::move(path), bytes = std::move(owned), cancellation, chunk = config_.chunk_size] {
        return write_sync(path, bytes, chunk, cancellation);
    });
}

job_future<file_result<void>> async_file_service::write_atomic(
    std::filesystem::path path,
    std::span<const std::byte> bytes,
    cancellation_token cancellation)
{
    file_buffer owned(bytes.begin(), bytes.end());
    return jobs_->submit_future({
        .name = "io.write_atomic",
        .priority = job_priority::high,
        .affinity = job_affinity::io_thread,
        .cancellation = cancellation
    }, [path = std::move(path), bytes = std::move(owned), cancellation, chunk = config_.chunk_size] {
        const auto temporary = temporary_path_for(path);
        auto written = write_sync(temporary, bytes, chunk, cancellation);
        if (!written)
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return written;
        }
        if (cancellation.stop_requested())
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return file_result<void>::failure(make_error(file_error_code::cancelled, path, "Atomic write was cancelled"));
        }
        if (!replace_file(temporary, path))
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return file_result<void>::failure(make_error(file_error_code::replace_failed, path, "Atomic destination replacement failed"));
        }
        return file_result<void>::success();
    });
}

job_future<file_result<file_info>> async_file_service::stat(
    std::filesystem::path path,
    cancellation_token cancellation)
{
    return jobs_->submit_future({
        .name = "io.stat",
        .priority = job_priority::normal,
        .affinity = job_affinity::io_thread,
        .cancellation = cancellation
    }, [path = std::move(path), cancellation] {
        if (cancellation.stop_requested())
            return file_result<file_info>::failure(make_error(file_error_code::cancelled, path, "Stat was cancelled"));
        std::error_code error;
        const bool regular = std::filesystem::is_regular_file(path, error);
        if (error || !regular)
            return file_result<file_info>::failure(open_error(path));
        const auto size = std::filesystem::file_size(path, error);
        if (error)
            return file_result<file_info>::failure(make_error(file_error_code::read_failed, path, "File size could not be read"));
        const auto modified = std::filesystem::last_write_time(path, error);
        if (error)
            return file_result<file_info>::failure(make_error(file_error_code::read_failed, path, "Modification time could not be read"));
        return file_result<file_info>::success({
            .size = size,
            .modified = modified,
            .regular_file = true
        });
    });
}

std::size_t async_file_service::chunk_size() const noexcept
{
    return config_.chunk_size;
}

job_system& async_file_service::scheduler() const noexcept
{
    return *jobs_;
}

} // namespace arc::io
