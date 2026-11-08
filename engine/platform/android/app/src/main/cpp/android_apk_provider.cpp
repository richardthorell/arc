#include "android_apk_provider.h"

#include <utility>

namespace arc::platform::android
{
namespace
{
arc::io::file_error error(arc::io::file_error_code code, std::string_view key, std::string_view message)
{
    return {.code = code, .message = std::string(message), .logical_path = std::string(key)};
}
} // namespace

arc::io::provider_capabilities android_apk_provider::capabilities() const noexcept
{
    return arc::io::provider_capability::full_read | arc::io::provider_capability::range_read |
           arc::io::provider_capability::metadata;
}

arc::io::provider_lookup_result android_apk_provider::resolve(std::string_view path)
{
    if (!valid_package_asset_path(path))
        return {.status = arc::io::provider_lookup_status::failure,
                .error = error(arc::io::file_error_code::invalid_request, path, "Invalid packaged asset path")};
    std::uint64_t size{};
    if (!source_->file_size(path, size)) return {.status = arc::io::provider_lookup_status::not_found};
    return {.status = arc::io::provider_lookup_status::found,
            .file = {.key = std::string(path), .size = size, .content_generation = 1}};
}

arc::jobs::job_future<arc::io::file_result<arc::io::file_buffer>>
android_apk_provider::read_range(const arc::io::provider_file& file, std::uint64_t offset, std::size_t bytes,
                                 arc::jobs::cancellation_token cancellation)
{
    // The source and AAssetManager outlive these reads; drain job_system before
    // destroying this provider during Android host shutdown.
    return jobs_->submit_future(
        {.name = "android.apk.read", .affinity = arc::jobs::job_affinity::io_thread, .cancellation = cancellation},
        [source = source_, file, offset, bytes, cancellation]
        {
            using arc::io::file_result;
            using arc::io::file_buffer;
            if (cancellation.stop_requested())
                return file_result<file_buffer>::failure(
                    error(arc::io::file_error_code::cancelled, file.key, "APK read cancelled"));
            if (file.content_generation != 1)
                return file_result<file_buffer>::failure(
                    error(arc::io::file_error_code::stale_handle, file.key, "Stale APK asset handle"));
            if (offset > file.size || bytes > file.size - offset)
                return file_result<file_buffer>::failure(
                    error(arc::io::file_error_code::invalid_range, file.key, "APK range exceeds asset size"));
            auto result = source->read_range(file.key, offset, bytes);
            if (!result)
            {
                const auto code = result.error == asset_read_error::unavailable ? arc::io::file_error_code::not_found
                                  : result.error == asset_read_error::invalid_range
                                      ? arc::io::file_error_code::invalid_range
                                      : arc::io::file_error_code::read_failed;
                return file_result<file_buffer>::failure(error(code, file.key, "APK asset read failed"));
            }
            return file_result<file_buffer>::success(std::move(result.bytes));
        });
}
} // namespace arc::platform::android
