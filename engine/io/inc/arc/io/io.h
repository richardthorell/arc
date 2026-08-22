#pragma once

/** @namespace arc::io
 * @brief Asynchronous, cancellable, tagged file operations.
 */

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
    file_error_code code{file_error_code::none};
    std::filesystem::path path;
    std::string message;
};

template <class T> class [[nodiscard]] file_result
{
public:
    [[nodiscard]] static file_result success(T value)
    {
        file_result result;
        result.value_.emplace(std::move(value));
        return result;
    }

    [[nodiscard]] static file_result failure(file_error error)
    {
        file_result result;
        result.error_ = std::move(error);
        return result;
    }

    [[nodiscard]] bool succeeded() const noexcept
    {
        return value_.has_value();
    }
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return succeeded();
    }
    [[nodiscard]] T& value() & noexcept
    {
        return *value_;
    }
    [[nodiscard]] const T& value() const& noexcept
    {
        return *value_;
    }
    [[nodiscard]] T&& value() && noexcept
    {
        return std::move(*value_);
    }
    [[nodiscard]] const file_error& error() const noexcept
    {
        return error_;
    }

private:
    std::optional<T> value_;
    file_error error_;
};

template <> class [[nodiscard]] file_result<void>
{
public:
    [[nodiscard]] static file_result success()
    {
        file_result result;
        result.succeeded_ = true;
        return result;
    }

    [[nodiscard]] static file_result failure(file_error error)
    {
        file_result result;
        result.error_ = std::move(error);
        return result;
    }

    [[nodiscard]] bool succeeded() const noexcept
    {
        return succeeded_;
    }
    [[nodiscard]] explicit operator bool() const noexcept
    {
        return succeeded();
    }
    [[nodiscard]] const file_error& error() const noexcept
    {
        return error_;
    }

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
    std::size_t chunk_size{1024u * 1024u};
};

class async_file_service
{
public:
    explicit async_file_service(jobs::job_system& jobs, async_file_config config = {});

    [[nodiscard]] jobs::job_future<file_result<file_buffer>> read_all(std::filesystem::path path,
                                                                      jobs::cancellation_token cancellation = {});
    [[nodiscard]] jobs::job_future<file_result<file_buffer>> read_range(std::filesystem::path path,
                                                                        std::uint64_t offset, std::size_t bytes,
                                                                        jobs::cancellation_token cancellation = {});
    [[nodiscard]] jobs::job_future<file_result<void>>
    write(std::filesystem::path path, std::span<const std::byte> bytes, jobs::cancellation_token cancellation = {});
    [[nodiscard]] jobs::job_future<file_result<void>> write_atomic(std::filesystem::path path,
                                                                   std::span<const std::byte> bytes,
                                                                   jobs::cancellation_token cancellation = {});
    [[nodiscard]] jobs::job_future<file_result<file_info>> stat(std::filesystem::path path,
                                                                jobs::cancellation_token cancellation = {});

    [[nodiscard]] std::size_t chunk_size() const noexcept;
    [[nodiscard]] jobs::job_system& scheduler() const noexcept;

private:
    jobs::job_system* jobs_{};
    async_file_config config_{};
};

} // namespace arc::io
