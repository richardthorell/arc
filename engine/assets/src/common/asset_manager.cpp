#include "asset_manager_internal.h"
#include <sqlite3.h>

#include <algorithm>
#include <mutex>
#include <unordered_set>

namespace arc::assets
{
using namespace manager_detail;

asset_pin::asset_pin(std::shared_ptr<detail::asset_slot> slot) : slot_(std::move(slot))
{
    if (slot_) slot_->pins.fetch_add(1, std::memory_order_relaxed);
}

asset_pin::~asset_pin()
{
    reset();
}

asset_pin::asset_pin(asset_pin&& other) noexcept : slot_(std::move(other.slot_)) {}

asset_pin& asset_pin::operator=(asset_pin&& other) noexcept
{
    if (this != &other)
    {
        reset();
        slot_ = std::move(other.slot_);
    }
    return *this;
}

void asset_pin::reset() noexcept
{
    if (slot_) slot_->pins.fetch_sub(1, std::memory_order_relaxed);
    slot_.reset();
}

asset_manager::asset_manager(asset_manager_config config, jobs::job_system& jobs, io::async_file_service& files,
                             memory::memory_system& memory)
    : implementation_(std::make_unique<implementation>(std::move(config), jobs, files, memory))
{
}

asset_manager::~asset_manager() = default;

void asset_manager::on_start(framework::runtime_service_context&)
{
    std::string error;
    {
        std::unique_lock lock(implementation_->mutex);
        if (implementation_->started) return;
        if (!implementation_->open_database(error))
        {
            const auto& original_error = error;
            std::string rebuild_error;
            if (implementation_->rebuild_database(rebuild_error))
                implementation_->events.push_back(
                    {.sequence = implementation_->next_event++,
                     .kind = asset_event_kind::discovered,
                     .message = "Rebuilt incompatible/corrupt asset registry: " + original_error});
            else
                implementation_->events.push_back({.sequence = implementation_->next_event++,
                                                   .kind = asset_event_kind::failed,
                                                   .message = "Asset registry database failed: " + original_error +
                                                              "; rebuild failed: " + rebuild_error});
        }
        implementation_->started = true;
        implementation_->next_poll = std::chrono::steady_clock::now() + implementation_->config.source_poll_interval;
    }
    (void)scan();
    implementation_->pressure_handler = implementation_->memory->add_pressure_handler(
        [this](memory::memory_pressure_level, memory::memory_domain domain, std::size_t)
        {
            if (domain == memory::memory_domain::assets || domain == memory::memory_domain::streaming ||
                domain == memory::memory_domain::general)
                evict_unused();
        });
}

void asset_manager::on_shutdown(framework::runtime_service_context&) noexcept
{
    std::vector<jobs::job_handle> active;
    {
        std::unique_lock lock(implementation_->mutex);
        if (!implementation_->started) return;
        implementation_->started = false;
        for (auto& [_, record] : implementation_->records)
        {
            record.import_cancellation.request_cancel();
            if (record.active_import.valid()) active.push_back(record.active_import);
        }
    }
    for (const auto& job : active)
        (void)job.wait_result();

    std::unique_lock lock(implementation_->mutex);
    if (implementation_->pressure_handler)
    {
        implementation_->memory->remove_pressure_handler(implementation_->pressure_handler);
        implementation_->pressure_handler = 0;
    }
    implementation_->subscribers.clear();
    if (implementation_->database)
    {
        sqlite3_close(implementation_->database);
        implementation_->database = nullptr;
    }
}

bool asset_manager::register_importer(std::unique_ptr<asset_importer> importer)
{
    if (!importer || !importer->descriptor().id.valid() || importer->descriptor().name.empty() ||
        importer->descriptor().version == 0 || importer->descriptor().output_types.empty())
        return false;
    std::unique_lock lock(implementation_->mutex);
    const auto id = importer->descriptor().id;
    implementation_->importers[id] = std::move(importer);
    return true;
}

bool asset_manager::register_virtual_asset(asset_guid guid, asset_type_id type, asset_payload payload, std::string name,
                                           bool pin_value)
{
    if (!guid.valid() || !type.valid() || payload.type() != type || !payload) return false;
    std::unique_lock lock(implementation_->mutex);
    if (implementation_->records.contains(guid)) return false;
    implementation::record value;
    value.virtual_asset = true;
    value.snapshot.guid = guid;
    value.snapshot.type = type;
    value.snapshot.source_path = "arc://builtin/" + name;
    value.snapshot.state = asset_state::ready;
    value.snapshot.residency = asset_residency::cpu;
    value.snapshot.generation = 1;
    value.snapshot.revision = ++implementation_->revision;
    value.snapshot.has_last_good = true;
    value.snapshot.read_only = true;
    value.slot->requested_guid = guid;
    value.slot->resolved_guid = guid;
    value.slot->type = type;
    value.slot->generation = 1;
    value.slot->payload = std::make_shared<const asset_payload>(std::move(payload));
    if (pin_value) value.slot->pins = 1;
    implementation_->paths[path_key(value.snapshot.source_path)] = guid;
    implementation_->records.emplace(guid, std::move(value));
    implementation_->emit(asset_event_kind::discovered, guid, asset_state::ready, "Built-in asset registered");
    return true;
}

bool asset_manager::set_fallback(asset_type_id type, asset_guid guid)
{
    std::unique_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(guid);
    if (!type.valid() || found == implementation_->records.end() || found->second.snapshot.type != type) return false;
    implementation_->fallbacks[type] = guid;
    return true;
}

asset_guid asset_manager::fallback_for(asset_type_id type) const noexcept
{
    std::shared_lock lock(implementation_->mutex);
    const auto found = implementation_->fallbacks.find(type);
    return found == implementation_->fallbacks.end() ? asset_guid{} : found->second;
}

std::vector<asset_importer_snapshot> asset_manager::importers() const
{
    std::shared_lock lock(implementation_->mutex);
    std::vector<asset_importer_snapshot> result;
    result.reserve(implementation_->importers.size());
    for (const auto& [_, importer] : implementation_->importers)
    {
        const auto& descriptor = importer->descriptor();
        result.push_back({.id = descriptor.id,
                          .name = descriptor.name,
                          .version = descriptor.version,
                          .settings_version = descriptor.settings_version,
                          .extensions = descriptor.extensions,
                          .output_types = descriptor.output_types});
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
    return result;
}

void asset_manager::poll()
{
    if (!implementation_->config.enable_source_monitor) return;
    const auto now = std::chrono::steady_clock::now();
    {
        std::shared_lock lock(implementation_->mutex);
        if (!implementation_->started || now < implementation_->next_poll) return;
    }
    {
        std::unique_lock lock(implementation_->mutex);
        implementation_->next_poll = now + implementation_->config.source_poll_interval;
    }
    (void)scan();
}

std::optional<asset_snapshot> asset_manager::find(asset_guid guid) const
{
    std::shared_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(guid);
    if (found == implementation_->records.end()) return std::nullopt;
    auto result = found->second.snapshot;
    result.strong_references =
        found->second.slot.use_count() > 0 ? static_cast<std::uint32_t>(found->second.slot.use_count() - 1) : 0;
    result.pins = found->second.slot->pins.load(std::memory_order_relaxed);
    return result;
}

std::optional<asset_snapshot> asset_manager::find(std::string_view project_relative_path) const
{
    std::shared_lock lock(implementation_->mutex);
    const auto path = implementation_->paths.find(path_key(std::string(project_relative_path)));
    if (path == implementation_->paths.end()) return std::nullopt;
    const auto found = implementation_->records.find(path->second);
    if (found == implementation_->records.end()) return std::nullopt;
    auto result = found->second.snapshot;
    result.strong_references =
        found->second.slot.use_count() > 0 ? static_cast<std::uint32_t>(found->second.slot.use_count() - 1) : 0;
    result.pins = found->second.slot->pins.load(std::memory_order_relaxed);
    return result;
}

std::vector<asset_snapshot> asset_manager::search(std::string_view text, std::optional<asset_type_id> type) const
{
    std::shared_lock lock(implementation_->mutex);
    std::string needle = path_key(std::string(text));
    std::vector<asset_snapshot> result;
    for (const auto& [_, value] : implementation_->records)
    {
        if (type && value.snapshot.type != *type) continue;
        if (!needle.empty() && path_key(value.snapshot.source_path).find(needle) == std::string::npos &&
            path_key(value.snapshot.title).find(needle) == std::string::npos &&
            path_key(value.snapshot.description).find(needle) == std::string::npos)
            continue;
        auto snapshot = value.snapshot;
        snapshot.strong_references =
            value.slot.use_count() > 0 ? static_cast<std::uint32_t>(value.slot.use_count() - 1) : 0;
        snapshot.pins = value.slot->pins.load(std::memory_order_relaxed);
        result.push_back(std::move(snapshot));
    }
    std::sort(result.begin(), result.end(), [](const auto& left, const auto& right)
              { return normalize_asset_path(left.source_path) < normalize_asset_path(right.source_path); });
    return result;
}

asset_registry_snapshot asset_manager::snapshot() const
{
    asset_registry_snapshot result;
    result.project_root = implementation_->config.project_root;
    result.asset_root = implementation_->config.asset_root;
    result.database_path = implementation_->config.cache_root / "assets.db";
    result.derived_data_root = implementation_->config.cache_root / "derived";
    {
        std::shared_lock lock(implementation_->mutex);
        result.revision = implementation_->revision;
        result.missing_references = implementation_->missing_references;
        result.assets.reserve(implementation_->records.size());
        for (const auto& [_, value] : implementation_->records)
        {
            auto snapshot = value.snapshot;
            snapshot.strong_references =
                value.slot.use_count() > 0 ? static_cast<std::uint32_t>(value.slot.use_count() - 1) : 0;
            snapshot.pins = value.slot->pins.load(std::memory_order_relaxed);
            result.assets.push_back(std::move(snapshot));
        }
    }
    std::sort(result.assets.begin(), result.assets.end(),
              [](const auto& left, const auto& right) { return left.guid < right.guid; });
    return result;
}

std::vector<asset_guid> asset_manager::dependencies(asset_guid guid) const
{
    const auto result = find(guid);
    return result ? result->dependencies : std::vector<asset_guid>{};
}

std::vector<asset_guid> asset_manager::reverse_dependencies(asset_guid guid) const
{
    const auto result = find(guid);
    return result ? result->reverse_dependencies : std::vector<asset_guid>{};
}

asset_reference asset_manager::resolve(std::string_view project_relative_path, asset_type_id expected_type) const
{
    asset_reference result{.expected_type = expected_type, .path_hint = std::string(project_relative_path)};
    const auto found = find(project_relative_path);
    if (!found || (expected_type.valid() && found->type != expected_type)) return result;
    result.guid = found->guid;
    if (!result.expected_type.valid()) result.expected_type = found->type;
    return result;
}

missing_asset_reference asset_manager::audit_reference(const asset_reference& reference, std::string owner,
                                                       std::string field)
{
    missing_asset_reference result{.reference = reference, .owner = std::move(owner), .field = std::move(field)};
    const auto found = find(reference.guid);
    if (found && !found->source_missing && (!reference.expected_type.valid() || reference.expected_type == found->type))
        return result;
    result.reason = !reference.guid.valid() ? "Reference has no asset GUID"
                    : !found                ? "Asset GUID is not registered"
                    : found->source_missing ? "Asset source is missing"
                                            : "Asset type does not match the reference";
    if (!reference.path_hint.empty())
    {
        if (const auto candidate = find(reference.path_hint)) result.repair_candidates.push_back(candidate->guid);
    }
    std::unique_lock lock(implementation_->mutex);
    implementation_->missing_references.push_back(result);
    ++implementation_->revision;
    implementation_->emit(asset_event_kind::missing_reference, reference.guid, asset_state::unknown, result.reason);
    return result;
}

bool asset_manager::set_dependencies(asset_guid guid, std::span<const asset_reference> dependencies_value)
{
    std::unique_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(guid);
    if (found == implementation_->records.end()) return false;
    std::vector<asset_guid> dependencies;
    dependencies.reserve(dependencies_value.size());
    for (const asset_reference& reference : dependencies_value)
    {
        if (!reference.guid.valid()) return false;
        std::unordered_set<asset_guid, asset_guid_hash> visited;
        if (implementation_->dependency_reaches(reference.guid, guid, visited)) return false;
        if (std::find(dependencies.begin(), dependencies.end(), reference.guid) == dependencies.end())
            dependencies.push_back(reference.guid);
    }
    for (asset_guid old : found->second.snapshot.dependencies)
    {
        if (const auto dependency = implementation_->records.find(old); dependency != implementation_->records.end())
            std::erase(dependency->second.snapshot.reverse_dependencies, guid);
    }
    found->second.snapshot.dependencies = dependencies;
    for (asset_guid dependency_guid : dependencies)
    {
        if (const auto dependency = implementation_->records.find(dependency_guid);
            dependency != implementation_->records.end() &&
            std::find(dependency->second.snapshot.reverse_dependencies.begin(),
                      dependency->second.snapshot.reverse_dependencies.end(),
                      guid) == dependency->second.snapshot.reverse_dependencies.end())
            dependency->second.snapshot.reverse_dependencies.push_back(guid);
    }
    std::vector<asset_hash> dependency_hashes;
    for (asset_guid dependency : dependencies)
        if (const auto dependency_record = implementation_->records.find(dependency);
            dependency_record != implementation_->records.end())
            dependency_hashes.push_back(!dependency_record->second.snapshot.dependency_hash.empty()
                                            ? dependency_record->second.snapshot.dependency_hash
                                            : dependency_record->second.snapshot.source_hash);
    dependency_hashes.insert(dependency_hashes.begin(), found->second.snapshot.source_hash);
    found->second.snapshot.dependency_hash = combine_hashes(dependency_hashes);
    found->second.snapshot.state = asset_state::stale;
    found->second.snapshot.revision = ++implementation_->revision;
    implementation_->persist_record(found->second);
    implementation_->persist_dependencies(found->second);
    implementation_->emit(asset_event_kind::dependencies_changed, guid, asset_state::stale,
                          "Asset dependencies changed");
    return true;
}

bool asset_manager::mark_stale(asset_guid guid, std::string reason)
{
    std::unique_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(guid);
    if (found == implementation_->records.end()) return false;
    found->second.snapshot.state = asset_state::stale;
    found->second.snapshot.revision = ++implementation_->revision;
    found->second.snapshot.diagnostics.push_back(
        implementation_->diagnostic(guid, asset_diagnostic_severity::information, "stale", std::move(reason)));
    implementation_->persist_record(found->second);
    implementation_->mark_reverse_stale(guid, "asset explicitly marked stale");
    implementation_->emit(asset_event_kind::state_changed, guid, asset_state::stale, "Asset marked stale");
    return true;
}

jobs::job_handle asset_manager::reimport(asset_guid guid, asset_streaming_priority priority,
                                         jobs::cancellation_token cancellation)
{
    return implementation_->ensure_import(guid, priority, cancellation);
}

bool asset_manager::cancel_import(asset_guid guid)
{
    std::shared_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(guid);
    return found != implementation_->records.end() && found->second.import_cancellation.request_cancel();
}

asset_move_result asset_manager::move(asset_guid guid, std::filesystem::path destination)
{
    asset_move_result result{.guid = guid};
    std::unique_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(guid);
    if (found != implementation_->records.end() && found->second.snapshot.read_only)
    {
        result.error = {.code = asset_error_code::invalid_request,
                        .guid = guid,
                        .path = found->second.snapshot.source_path,
                        .message = "Built-in assets are read-only"};
        return result;
    }
    if (found == implementation_->records.end() || found->second.virtual_asset)
    {
        result.error = {.code = asset_error_code::not_found, .guid = guid, .message = "Asset is not movable"};
        return result;
    }
    if (destination.is_relative()) destination = implementation_->config.project_root / destination;
    destination = std::filesystem::absolute(destination).lexically_normal();
    if (!implementation_->managed_source_path(destination) ||
        classify_asset_path(destination) != classify_asset_path(found->second.absolute_path))
    {
        result.error = {.code = asset_error_code::invalid_request,
                        .guid = guid,
                        .path = destination,
                        .message =
                            "Asset destination must remain inside a managed project source root and preserve its type"};
        return result;
    }
    result.previous_path = found->second.snapshot.source_path;
    const auto previous_absolute = found->second.absolute_path;
    const auto previous_metadata = metadata_path_for(previous_absolute);
    const auto destination_metadata = metadata_path_for(destination);
    std::error_code error;
    std::filesystem::create_directories(destination.parent_path(), error);
    if (error || std::filesystem::exists(destination))
    {
        result.error = {.code = asset_error_code::io_failed,
                        .guid = guid,
                        .path = destination,
                        .message = error ? error.message() : "Asset destination already exists"};
        return result;
    }
    std::filesystem::rename(previous_absolute, destination, error);
    if (error)
    {
        result.error = {
            .code = asset_error_code::io_failed, .guid = guid, .path = destination, .message = error.message()};
        return result;
    }
    std::filesystem::rename(previous_metadata, destination_metadata, error);
    if (error)
    {
        std::error_code rollback;
        std::filesystem::rename(destination, previous_absolute, rollback);
        result.error = {.code = asset_error_code::io_failed,
                        .guid = guid,
                        .path = destination_metadata,
                        .message = "Could not move asset metadata; source move was rolled back"};
        return result;
    }
    implementation_->paths.erase(path_key(found->second.snapshot.source_path));
    found->second.absolute_path = destination;
    found->second.snapshot.source_path = destination.lexically_relative(implementation_->config.project_root);
    found->second.snapshot.revision = ++implementation_->revision;
    implementation_->paths[path_key(found->second.snapshot.source_path)] = guid;
    implementation_->persist_record(found->second);
    result.current_path = found->second.snapshot.source_path;
    implementation_->emit(asset_event_kind::moved, guid, found->second.snapshot.state, "Asset moved");
    return result;
}

asset_move_result asset_manager::rename(asset_guid guid, std::string filename)
{
    if (filename.empty() || std::filesystem::path(filename).filename().string() != filename)
        return {.guid = guid,
                .error = {.code = asset_error_code::invalid_request,
                          .guid = guid,
                          .message = "Asset filename must not contain a directory"}};
    const auto found = find(guid);
    if (!found)
        return {.guid = guid,
                .error = {.code = asset_error_code::not_found, .guid = guid, .message = "Asset was not found"}};
    return move(guid, found->source_path.parent_path() / filename);
}

jobs::job_future<asset_manager::untyped_load_result> asset_manager::load_untyped(asset_load_request request)
{
    asset_guid guid = request.reference.guid;
    asset_error immediate_error;
    std::shared_ptr<detail::asset_slot> immediate_slot;
    {
        std::shared_lock lock(implementation_->mutex);
        const auto found = implementation_->records.find(guid);
        if (found == implementation_->records.end() || found->second.snapshot.source_missing)
        {
            immediate_error = {
                .code = asset_error_code::not_found, .guid = guid, .message = "Asset reference could not be resolved"};
            if (request.allow_fallback)
            {
                const auto fallback = implementation_->fallbacks.find(request.reference.expected_type);
                if (fallback != implementation_->fallbacks.end())
                    if (const auto fallback_record = implementation_->records.find(fallback->second);
                        fallback_record != implementation_->records.end())
                        immediate_slot = fallback_record->second.slot;
            }
        }
        else if (request.reference.expected_type.valid() &&
                 request.reference.expected_type != found->second.snapshot.type)
        {
            immediate_error = {.code = asset_error_code::type_mismatch,
                               .guid = guid,
                               .message = "Asset type does not match the reference"};
        }
        else if (found->second.slot->payload.load(std::memory_order_acquire))
        {
            immediate_slot = found->second.slot;
        }
    }
    if (immediate_slot || immediate_error)
    {
        return implementation_->jobs->submit_future(
            {.name = "assets.load.immediate", .priority = to_job_priority(request.priority)},
            [slot = std::move(immediate_slot), error = std::move(immediate_error), cancellation = request.cancellation]
            {
                if (cancellation.stop_requested())
                    return untyped_load_result{.error = {.code = asset_error_code::cancelled,
                                                         .guid = slot ? slot->requested_guid : error.guid,
                                                         .message = "Asset load was cancelled"}};
                return untyped_load_result{.slot = slot, .error = error};
            });
    }

    // A request token cancels only this waiter. The shared import generation is
    // cancelled explicitly through cancel_import(), never by one of several clients.
    const jobs::job_handle imported = implementation_->ensure_import(guid, request.priority, {}, request.residency);
    return implementation_->jobs->submit_future(
        {.name = "assets.load.complete",
         .priority = to_job_priority(request.priority),
         .dependencies = imported.valid() ? std::vector<jobs::job_handle>{imported} : std::vector<jobs::job_handle>{},
         .dependency_policy = jobs::job_dependency_policy::run_always},
        [this, guid, request]
        {
            if (request.cancellation.stop_requested())
                return untyped_load_result{.error = {.code = asset_error_code::cancelled,
                                                     .guid = guid,
                                                     .message = "Asset load was cancelled"}};
            std::shared_lock lock(implementation_->mutex);
            const auto found = implementation_->records.find(guid);
            if (found != implementation_->records.end() && found->second.slot->payload.load(std::memory_order_acquire))
            {
                asset_error error;
                if (found->second.snapshot.state == asset_state::failed)
                {
                    error = {.code = asset_error_code::import_failed,
                             .guid = guid,
                             .path = found->second.snapshot.source_path,
                             .message = "Asset import failed; using the last-good generation"};
                }
                return untyped_load_result{.slot = found->second.slot, .error = std::move(error)};
            }
            if (request.allow_fallback)
            {
                const auto fallback = implementation_->fallbacks.find(request.reference.expected_type);
                if (fallback != implementation_->fallbacks.end())
                    if (const auto value = implementation_->records.find(fallback->second);
                        value != implementation_->records.end())
                        return untyped_load_result{
                            .slot = value->second.slot,
                            .error = {.code = asset_error_code::import_failed,
                                      .guid = guid,
                                      .message = "Asset import failed; using an explicit fallback"}};
            }
            return untyped_load_result{
                .error = {.code = request.cancellation.stop_requested() ? asset_error_code::cancelled
                                                                        : asset_error_code::import_failed,
                          .guid = guid,
                          .message = request.cancellation.stop_requested() ? "Asset load was cancelled"
                                                                           : "Asset failed to load"}};
        });
}

jobs::job_handle asset_manager::prefetch(asset_load_request request)
{
    return load_untyped(std::move(request)).handle();
}

asset_pin asset_manager::pin(asset_guid guid)
{
    std::shared_lock lock(implementation_->mutex);
    const auto found = implementation_->records.find(guid);
    return found == implementation_->records.end() ? asset_pin{} : asset_pin(found->second.slot);
}

std::size_t asset_manager::evict_unused(asset_residency maximum_residency)
{
    std::unique_lock lock(implementation_->mutex);
    std::vector<implementation::record*> candidates;
    for (auto& [_, value] : implementation_->records)
    {
        if (value.virtual_asset || value.slot.use_count() != 1 ||
            value.slot->pins.load(std::memory_order_relaxed) != 0 || value.snapshot.residency > maximum_residency ||
            !value.slot->payload.load(std::memory_order_acquire))
            continue;
        candidates.push_back(&value);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto* left, const auto* right)
              {
                  if (left->snapshot.residency != right->snapshot.residency)
                      return left->snapshot.residency > right->snapshot.residency;
                  return left->last_used < right->last_used;
              });
    for (auto* value : candidates)
    {
        value->slot->payload.store(nullptr, std::memory_order_release);
        value->snapshot.residency = asset_residency::derived;
        value->snapshot.revision = ++implementation_->revision;
        implementation_->persist_record(*value);
        implementation_->emit(asset_event_kind::evicted, value->snapshot.guid, value->snapshot.state,
                              "Unused asset payload evicted");
    }
    return candidates.size();
}

std::uint64_t asset_manager::subscribe(asset_event_callback callback)
{
    if (!callback) return 0;
    std::unique_lock lock(implementation_->mutex);
    const auto token = implementation_->next_subscription++;
    implementation_->subscribers.emplace(token, std::move(callback));
    return token;
}

bool asset_manager::unsubscribe(std::uint64_t token)
{
    std::unique_lock lock(implementation_->mutex);
    return implementation_->subscribers.erase(token) != 0;
}

std::vector<asset_event> asset_manager::events_since(std::uint64_t sequence) const
{
    std::shared_lock lock(implementation_->mutex);
    std::vector<asset_event> result;
    std::copy_if(implementation_->events.begin(), implementation_->events.end(), std::back_inserter(result),
                 [sequence](const asset_event& event) { return event.sequence > sequence; });
    return result;
}

const asset_manager_config& asset_manager::config() const noexcept
{
    return implementation_->config;
}
jobs::job_system& asset_manager::jobs() const noexcept
{
    return *implementation_->jobs;
}

jobs::job_priority asset_manager::to_job_priority(asset_streaming_priority priority) noexcept
{
    switch (priority)
    {
        case asset_streaming_priority::background:
            return jobs::job_priority::background;
        case asset_streaming_priority::low:
            return jobs::job_priority::low;
        case asset_streaming_priority::normal:
            return jobs::job_priority::normal;
        case asset_streaming_priority::high:
            return jobs::job_priority::high;
        case asset_streaming_priority::critical:
            return jobs::job_priority::critical;
    }
    return jobs::job_priority::normal;
}

} // namespace arc::assets
