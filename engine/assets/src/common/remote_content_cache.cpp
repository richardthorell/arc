#include <arc/assets/remote_content_cache.h>

#include <span>
#include <string>
#include <utility>

namespace arc::assets
{
namespace
{

remote_object_cache_result failed(remote_object_cache_error code, std::string message)
{
    return {.code = code, .message = std::move(message)};
}

std::optional<content_hash> validated_hash(const remote_content_object& object)
{
    if (object.size == 0 || object.path != "objects/sha256/" + object.sha256)
        return std::nullopt;
    auto hash = parse_asset_hash(object.sha256);
    if (!hash || to_string(*hash) != object.sha256) return std::nullopt;
    return hash;
}

} // namespace

std::optional<io::file_buffer> load_verified_remote_object(derived_data_cache& cache,
                                                            const remote_content_object& object,
                                                            cache_error& error)
{
    error = {};
    const auto expected_hash = validated_hash(object);
    if (!expected_hash)
    {
        error.message = "Invalid immutable cooked-object SHA-256 reference";
        return std::nullopt;
    }
    auto cached = cache.get_blob(*expected_hash, error);
    if (!cached) return std::nullopt;
    if (cached->bytes.size() != object.size || hash_bytes(cached->bytes) != *expected_hash)
    {
        error.message = "Cached cooked object has mismatched length or SHA-256";
        return std::nullopt;
    }
    return std::move(cached->bytes);
}

remote_object_cache_result ingest_verified_remote_object(derived_data_cache& cache,
                                                         const remote_content_object& object,
                                                         const io::content_http_request& request,
                                                         const io::content_http_response& response,
                                                         const io::content_http_limits& limits)
{
    const auto expected_hash = validated_hash(object);
    if (!expected_hash || request.object_path != object.path || request.object_size != object.size ||
        request.range_start || request.range_length)
        return failed(remote_object_cache_error::invalid_reference,
                      "Remote CAS ingestion requires a matching complete immutable object");

    const auto validated = io::validate_content_http_response(request, response, limits);
    if (!validated.succeeded())
        return failed(remote_object_cache_error::incompatible_response, validated.message);

    if (hash_bytes(response.body) != *expected_hash)
        return failed(remote_object_cache_error::hash_mismatch,
                      "Downloaded cooked-object bytes do not match declared SHA-256");

    cache_error error;
    if (auto existing = load_verified_remote_object(cache, object, error))
        return {.code = remote_object_cache_error::none, .message = {}, .reused = true};

    // A corrupt old entry is never trusted; the cache must replace/reject it
    // rather than treating a stale hash-keyed file as a successful hit.
    if (!cache.put_blob(*expected_hash, std::span<const std::byte>(response.body), error))
        return failed(remote_object_cache_error::cache_failure,
                      error.message.empty() ? "Persistent cooked-object cache write failed" : error.message);

    auto persisted = load_verified_remote_object(cache, object, error);
    if (!persisted)
        return failed(remote_object_cache_error::cache_failure,
                      error.message.empty() ? "Persistent cooked-object readback verification failed" : error.message);
    return {};
}

} // namespace arc::assets
