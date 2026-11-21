#pragma once

/** @file
 * @brief C9.2c verified immutable HTTP cooked-object ingestion into ARC's CAS.
 *
 * The existing cook manifest, GUID catalog and runtime update receiver remain
 * authoritative. This boundary does not mount content or publish a generation.
 */

#include <arc/assets/cook.h>
#include <arc/assets/remote_content.h>
#include <arc/io/content_http.h>

#include <cstdint>
#include <optional>
#include <string>

namespace arc::assets
{

enum class remote_object_cache_error : std::uint8_t
{
    none,
    invalid_reference,
    incompatible_response,
    hash_mismatch,
    cache_failure
};

struct [[nodiscard]] remote_object_cache_result
{
    remote_object_cache_error code{remote_object_cache_error::none};
    std::string message;
    bool reused{};

    [[nodiscard]] bool succeeded() const noexcept
    {
        return code == remote_object_cache_error::none;
    }
};

/** Reuse a previously verified immutable object, or return nullopt on miss.
 * Corrupt entries must not be returned as valid content.
 * The cache's own error is preserved in 'error' for diagnostics.
 */
[[nodiscard]] std::optional<io::file_buffer>
load_verified_remote_object(derived_data_cache& cache, const remote_content_object& object, cache_error& error);

/** Validate full HTTP response framing, hash, and bytes BEFORE publishing into
 * the existing persistent derived-data CAS. A network worker calls this only
 * after fetch_content_http has completed; no render/main thread blocking.
 *
 * Identical content already in cache is reused, not rewritten.
 */
[[nodiscard]] remote_object_cache_result ingest_verified_remote_object(derived_data_cache& cache,
                                                                       const remote_content_object& object,
                                                                       const io::content_http_request& request,
                                                                       const io::content_http_response& response,
                                                                       const io::content_http_limits& limits);

} // namespace arc::assets
