#pragma once

#include <arc/jobs/jobs.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace arc::io
{

enum class file_error_code : std::uint8_t
{
    none,
    not_found,
    permission_denied,
    invalid_range,
    cancelled,
    read_failed,
    write_failed,
    replace_failed,
    invalid_request
};

struct file_error
{
    file_error_code code{ file_error_code::none };
    std::filesystem::path path;
    std::string message;
};

template <class T>
class file_result
{
public:
    static file_result success(T value)
    {
        file_result result;
        result.value_.emplace(std::move(value));
        return result;
    }

    static file_result failure(file_error error)
    {
        file_result result;
        result.error_ = std::move(error);
        return result;
    }

    bool succeeded() const noexcept { return value_.has_value(); }
    explicit operator bool() const noexcept { return succeeded(); }
    T& value() & { return *value_; }
    const T& value() const& { return *value_; }
    T&& value() && { return std::move(*value_); }
    const file_error& error() const noexcept { return error_; }

private:
    std::optional<T> value_;
    file_error error_;
};

template <>
class file_result<void>
{
public:
    static file_result success()
    {
        file_result result;
        result.succeeded_ = true;
        return result;
    }

    static file_result failure(file_error error)
    {
        file_result result;
        result.error_ = std::move(error);
        return result;
    }

    bool succeeded() const noexcept { return succeeded_; }
    explicit operator bool() const noexcept { return succeeded(); }
    const file_error& error() const noexcept { return error_; }

private:
    bool succeeded_{};
    file_error error_;
};

using file_buffer = std::vector<std::byte>;

struct file_info
{
    std::uint64_t size{};
    std::filesystem::file_time_type modified{};
    bool regular_file{};
};

struct async_file_config
{
    std::size_t chunk_size{ 1024u * 1024u };
};

class async_file_service
{
public:
    explicit async_file_service(job_system& jobs, async_file_config config = {});

    job_future<file_result<file_buffer>> read_all(
        std::filesystem::path path,
        cancellation_token cancellation = {});
    job_future<file_result<file_buffer>> read_range(
        std::filesystem::path path,
        std::uint64_t offset,
        std::size_t bytes,
        cancellation_token cancellation = {});
    job_future<file_result<void>> write(
        std::filesystem::path path,
        std::span<const std::byte> bytes,
        cancellation_token cancellation = {});
    job_future<file_result<void>> write_atomic(
        std::filesystem::path path,
        std::span<const std::byte> bytes,
        cancellation_token cancellation = {});
    job_future<file_result<file_info>> stat(
        std::filesystem::path path,
        cancellation_token cancellation = {});

    std::size_t chunk_size() const noexcept;
    job_system& scheduler() const noexcept;

private:
    job_system* jobs_{};
    async_file_config config_{};
};

} // namespace arc::io
