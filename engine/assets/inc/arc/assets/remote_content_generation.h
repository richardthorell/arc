#pragma once

/** @file
 * @brief C9.2c2 verified complete remote cook generations backed by ARC's existing CAS.
 *
 * A remote provider is constructible only after the cook manifest, every
 * declared package chunk and every artifact byte range have been verified.
 * The existing cooked Asset Database/VFS remains authoritative.
 */

#include <arc/assets/remote_content_cache.h>
#include <arc/assets/runtime_content.h>

#include <cstdint>
#include <memory>
#include <functional>
#include <string>

namespace arc::assets
{

/** A single immutable chunk fetch, injectable for native VFS tests. */
using remote_generation_fetch_executor = std::function<jobs::job_future<http::content_http_fetch_result>(
    jobs::job_system&, http::content_http_request, http::content_http_limits, jobs::cancellation_token)>;

/** Lazy mode requires a verified cached cook manifest, not cached chunks. */
struct lazy_remote_content_options
{
    std::string origin;
    http::content_http_limits http{};
    bool offline_only{};
    std::size_t maximum_chunks{100000};
    std::uint64_t maximum_generation_bytes{16ull * 1024ull * 1024ull * 1024ull};
};

class remote_content_artifact_provider final : public io::virtual_file_provider
{
public:
    using create_result = core::result<std::shared_ptr<remote_content_artifact_provider>, asset_error>;

    /** This synchronous verification step belongs on a worker thread after HTTP
     * ingestion. Cache and async file service must outlive the mounted provider.
     * Missing/corrupt objects are rejected before the provider can be mounted.
     */
    [[nodiscard]] static create_result create(const remote_content_release& release, derived_data_cache& cache,
                                              io::async_file_service& files);
    /** Metadata-only mount: the SHA-verified cook manifest must already be in
     * CAS. This validates the entire chunk declaration/range layout against
     * descriptor sizes, but fetches no chunk until a VFS read needs it.
     * A downloaded chunk is never served without whole-object SHA-256 and
     * addressed artifact-hash validation. Failed reads do not publish bytes.
     * files, cache, jobs and executor dependencies must outlive the provider.
     */
    [[nodiscard]] static create_result create_lazy(const remote_content_release& release, derived_data_cache& cache,
                                                   io::async_file_service& files, lazy_remote_content_options options,
                                                   remote_generation_fetch_executor executor = {});
    ~remote_content_artifact_provider() override;

    remote_content_artifact_provider(const remote_content_artifact_provider&) = delete;
    remote_content_artifact_provider& operator=(const remote_content_artifact_provider&) = delete;

    [[nodiscard]] io::provider_capabilities capabilities() const noexcept override;
    [[nodiscard]] std::uint64_t provider_generation() const noexcept override;
    [[nodiscard]] io::provider_lookup_result resolve(std::string_view relative_path) override;
    [[nodiscard]] jobs::job_future<io::file_result<io::file_buffer>>
    read_range(const io::provider_file& file, std::uint64_t offset, std::size_t bytes,
               jobs::cancellation_token cancellation = {}) override;
    [[nodiscard]] io::file_result<std::vector<io::provider_directory_entry>>
    enumerate(std::string_view relative_prefix) override;

    [[nodiscard]] const cook_manifest& manifest() const noexcept;

private:
    [[nodiscard]] static create_result create_impl(const remote_content_release& release, derived_data_cache& cache,
                                                   io::async_file_service& files,
                                                   std::optional<lazy_remote_content_options> lazy,
                                                   remote_generation_fetch_executor executor);
    struct implementation;
    explicit remote_content_artifact_provider(std::shared_ptr<implementation> state);
    std::shared_ptr<implementation> implementation_;
};

} // namespace arc::assets
