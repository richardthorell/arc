#include <arc/assets/remote_runtime_generation.h>

#include <limits>
#include <utility>

namespace arc::assets
{
namespace
{
constexpr std::int32_t overlay_priority = 1000000000;

asset_status reject(asset_error_code code, std::string message)
{
    return asset_status::failure({.code = code, .message = std::move(message)});
}

bool matching_target(const remote_content_target& a, const remote_content_target& b) noexcept
{
    return a.platform == b.platform && a.architecture == b.architecture && a.renderer == b.renderer &&
           a.cook_profile == b.cook_profile && a.texture_family == b.texture_family;
}

} // namespace

remote_runtime_generation_session::remote_runtime_generation_session(
    cooked_asset_catalog& catalog, derived_data_cache& cache, io::async_file_service& files,
    remote_content_expectations expected, std::int32_t initial_priority)
    : catalog_(&catalog), cache_(&cache), files_(&files), expected_(std::move(expected)),
      next_priority_(initial_priority)
{
    highest_accepted_ = expected_.minimum_generation;
}

remote_runtime_generation_session::~remote_runtime_generation_session() = default;

bool remote_runtime_generation_session::compatible(const remote_content_release& release) const noexcept
{
    return release.project_guid == expected_.project_guid &&
           release.engine_version == expected_.engine_version &&
           release.base_build_id == expected_.base_build_id && matching_target(release.target, expected_.target);
}

asset_status remote_runtime_generation_session::activate_verified(const remote_content_release& candidate)
{
    const auto fail = [&](asset_error_code code, std::string message)
    {
        ++rejected_activations_;
        return reject(code, std::move(message));
    };
    if (!compatible(candidate))
        return fail(asset_error_code::invalid_metadata, "Remote cooked generation does not match runtime identity");
    if (candidate.generation <= highest_accepted_ || candidate.generation == 0)
        return fail(asset_error_code::invalid_metadata, "Remote cooked generation is stale or regressed");
    if (next_priority_ < 1 || next_priority_ >= overlay_priority - 1)
        return fail(asset_error_code::invalid_request, "Remote provider priority exhausted or overlaps live overlay");

    // The *strict* factory checks the SHA-verified cook manifest, all package
    // chunks and each addressed artifact before exposing any VFS mount.
    auto prepared = remote_content_artifact_provider::create(candidate, *cache_, *files_);
    if (!prepared)
    {
        ++rejected_activations_;
        return asset_status::failure(std::move(prepared).error());
    }

    // Persist active+last-good retention before exposing the new generation.
    // A failure never replaces the existing provider. Retention metadata
    // deliberately does not imply that the generation is trusted or mounted.
    auto retained = retain_remote_content_generations(*cache_, candidate, active_);
    if (!retained.succeeded()) return fail(asset_error_code::io_failed, retained.error);

    std::string new_provider_id;
    const auto mounted = catalog_->mount_remote_provider(std::move(prepared).value(), next_priority_,
                                                          &new_provider_id);
    if (!mounted)
    {
        // Restoring old pins may fail due to disk error, but over-retention is
        // safer than evicting the current generation; never silently promote.
        if (active_)
        {
            const auto restored = retain_remote_content_generations(*cache_, *active_, last_good_);
            (void)restored;
        }
        else
        {
            cache_error ignored;
            (void)clear_remote_content_retention(*cache_, candidate, ignored);
        }
        ++rejected_activations_;
        return mounted;
    }

    // Mount is visible now. Retire the oldest remote base, retaining the
    // previous active as last-good. The current candidate is always at a
    // strictly higher VFS priority than either prior provider.
    const auto stale_provider_id = last_good_provider_id_;
    last_good_ = std::move(active_);
    last_good_provider_id_ = std::move(active_provider_id_);
    active_ = candidate;
    active_provider_id_ = std::move(new_provider_id);
    highest_accepted_ = candidate.generation;
    ++next_priority_;

    if (!stale_provider_id.empty())
    {
        // A failure may leave an older, lower-priority provider mounted for
        // diagnostics, but cannot overwrite the current or previous base.
        if (!catalog_->unmount_provider(stale_provider_id)) ++cleanup_failures_;
    }
    return asset_status::success();
}

asset_status remote_runtime_generation_session::rollback()
{
    if (!active_ || !last_good_)
        return reject(asset_error_code::invalid_request, "No verified last-good remote generation to restore");
    if (overlay_ && overlay_->active_sequence() != 0)
        return reject(asset_error_code::invalid_request,
                      "Cannot roll back a base generation while a live artifact overlay is active");

    const auto retained = retain_remote_content_generations(*cache_, *last_good_);
    if (!retained.succeeded()) return reject(asset_error_code::io_failed, retained.error);

    const auto removed = catalog_->unmount_provider(active_provider_id_);
    if (!removed)
    {
        // The current provider is still live: retain both generations.
        const auto restored = retain_remote_content_generations(*cache_, *active_, last_good_);
        (void)restored;
        return removed;
    }
    active_ = std::move(last_good_);
    active_provider_id_ = std::move(last_good_provider_id_);
    last_good_.reset();
    last_good_provider_id_.clear();
    // highest_accepted_ intentionally stays monotonic across rollback:
    // callers cannot replay a stale/bad generation without a newer release.
    return asset_status::success();
}

asset_status remote_runtime_generation_session::attach_overlay(std::filesystem::path root)
{
    if (!active_)
        return reject(asset_error_code::invalid_request, "Cannot attach a live overlay before base activation");
    if (overlay_)
        return reject(asset_error_code::invalid_request, "Runtime live overlay already attached");
    if (root.empty())
        return reject(asset_error_code::invalid_request, "Runtime live overlay requires an explicit private cache root");

    auto overlay = std::make_shared<cas_overlay_provider>(*files_, root);
    const auto restored = overlay->load(expected_.target.cook_profile, expected_.base_build_id);
    if (!restored) return restored;
    const auto mounted = catalog_->mount_overlay(overlay, overlay_priority);
    if (!mounted) return mounted;

    receiver_ = std::make_unique<runtime_update_receiver>(
        overlay, runtime_update_receiver_config{.cache_root = std::move(root),
                                                .target_profile = expected_.target.cook_profile,
                                                .base_build_id = expected_.base_build_id});
    overlay_ = std::move(overlay);
    return asset_status::success();
}

asset_status remote_runtime_generation_session::ingest_overlay(
    const runtime_update_manifest& update, std::span<const std::pair<content_hash, io::file_buffer>> blobs)
{
    if (!active_ || !receiver_)
        return reject(asset_error_code::invalid_request, "Remote base and live overlay are not active");
    return receiver_->ingest(update, blobs);
}

remote_runtime_generation_state remote_runtime_generation_session::state() const
{
    return {.active_generation = active_ ? active_->generation : 0,
            .last_good_generation = last_good_ ? last_good_->generation : 0,
            .highest_accepted_generation = highest_accepted_,
            .overlay_sequence = overlay_ ? overlay_->active_sequence() : 0,
            .rejected_activations = rejected_activations_,
            .retired_mount_cleanup_failures = cleanup_failures_,
            .overlay_attached = overlay_ != nullptr,
            .active_provider_id = active_provider_id_};
}

} // namespace arc::assets
