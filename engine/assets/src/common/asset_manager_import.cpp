#include "asset_manager_internal.h"
#include <sqlite3.h>

#include <algorithm>
#include <mutex>
#include <unordered_set>

namespace arc::assets
{
using namespace manager_detail;

bool asset_manager::implementation::dependency_reaches(asset_guid current, asset_guid target,
                                                       std::unordered_set<asset_guid, asset_guid_hash>& visited) const
{
    if (current == target) return true;
    if (!visited.insert(current).second) return false;
    const auto found = records.find(current);
    if (found == records.end()) return false;
    return std::any_of(found->second.snapshot.dependencies.begin(), found->second.snapshot.dependencies.end(),
                       [&](asset_guid dependency) { return dependency_reaches(dependency, target, visited); });
}

void asset_manager::implementation::mark_reverse_stale(asset_guid guid, std::string_view reason)
{
    const auto found = records.find(guid);
    if (found == records.end()) return;
    for (asset_guid dependent : found->second.snapshot.reverse_dependencies)
    {
        const auto dependent_found = records.find(dependent);
        if (dependent_found == records.end() || dependent_found->second.snapshot.state == asset_state::stale) continue;
        dependent_found->second.snapshot.state = asset_state::stale;
        dependent_found->second.snapshot.revision = ++revision;
        dependent_found->second.snapshot.diagnostics.push_back(
            diagnostic(dependent, asset_diagnostic_severity::information, "dependency",
                       "Dependency changed: " + std::string(reason)));
        persist_record(dependent_found->second);
        emit(asset_event_kind::dependencies_changed, dependent, asset_state::stale, "Asset dependency became stale");
        mark_reverse_stale(dependent, reason);
    }
}

jobs::job_handle asset_manager::implementation::ensure_import(asset_guid guid, asset_streaming_priority priority,
                                                              jobs::cancellation_token cancellation,
                                                              asset_residency requested_residency)
{
    std::unique_lock lock(mutex);
    const auto found = records.find(guid);
    if (found == records.end() || found->second.virtual_asset) return {};
    record& value = found->second;
    value.requested_residency = std::max(value.requested_residency, requested_residency);
    if (value.active_import.valid() && !value.active_import.ready()) return value.active_import;

    const auto importer_found = importers.find(value.snapshot.importer);
    if (importer_found == importers.end())
    {
        value.snapshot.state = asset_state::failed;
        value.snapshot.revision = ++revision;
        value.snapshot.diagnostics.push_back(
            diagnostic(guid, asset_diagnostic_severity::error, "import", "No importer is registered for this asset"));
        persist_record(value);
        emit(asset_event_kind::failed, guid, asset_state::failed, "No importer is registered");
        return {};
    }

    value.import_cancellation = jobs::cancellation_source{};
    value.snapshot.state = asset_state::queued;
    value.snapshot.revision = ++revision;
    persist_record(value);
    emit(asset_event_kind::state_changed, guid, asset_state::queued, "Asset import queued");
    const auto importer_id = value.snapshot.importer;
    const auto affinity = importer_found->second->descriptor().affinity;
    value.active_import = jobs->submit(
        {.name = "assets.import",
         .priority = asset_manager::to_job_priority(priority),
         .affinity = affinity,
         .cancellation = cancellation.valid() ? cancellation : value.import_cancellation.token()},
        [this, guid, importer_id, priority, cancellation]
        {
            std::filesystem::path source_path;
            std::string source_path_hint;
            asset_source_metadata metadata;
            asset_hash source_hash;
            asset_residency import_residency{asset_residency::cpu};
            std::vector<asset_guid> dependencies;
            jobs::cancellation_token effective_cancellation = cancellation;
            {
                std::unique_lock state_lock(mutex);
                const auto current = records.find(guid);
                if (current == records.end()) return;
                current->second.snapshot.state = asset_state::importing;
                current->second.snapshot.revision = ++revision;
                source_path = current->second.absolute_path;
                source_path_hint = normalize_asset_path(current->second.snapshot.source_path);
                metadata = current->second.metadata;
                source_hash = current->second.snapshot.source_hash;
                import_residency = current->second.requested_residency;
                dependencies = current->second.snapshot.dependencies;
                if (!effective_cancellation.valid())
                    effective_cancellation = current->second.import_cancellation.token();
                persist_record(current->second);
                emit(asset_event_kind::state_changed, guid, asset_state::importing, "Asset import started");
            }

            for (asset_guid dependency : dependencies)
            {
                if (effective_cancellation.stop_requested()) break;
                const auto dependency_job = ensure_import(dependency, priority, {});
                if (dependency_job.valid())
                {
                    const auto result = dependency_job.wait_result();
                    if (!result.succeeded())
                    {
                        std::unique_lock state_lock(mutex);
                        if (const auto current = records.find(guid); current != records.end())
                        {
                            current->second.snapshot.state = asset_state::failed;
                            current->second.snapshot.revision = ++revision;
                            current->second.snapshot.diagnostics.push_back(
                                diagnostic(guid, asset_diagnostic_severity::error, "dependency",
                                           "An asset dependency failed to import"));
                            persist_record(current->second);
                            emit(asset_event_kind::failed, guid, asset_state::failed, "Dependency import failed");
                        }
                        return;
                    }
                }
            }

            if (effective_cancellation.stop_requested())
            {
                std::unique_lock state_lock(mutex);
                if (const auto current = records.find(guid); current != records.end())
                {
                    current->second.snapshot.state = asset_state::stale;
                    current->second.snapshot.revision = ++revision;
                    persist_record(current->second);
                    emit(asset_event_kind::state_changed, guid, asset_state::stale, "Asset import cancelled");
                }
                return;
            }

            auto read = files->read_all(source_path, effective_cancellation).get();
            if (!read)
            {
                std::unique_lock state_lock(mutex);
                if (const auto current = records.find(guid); current != records.end())
                {
                    current->second.snapshot.state = asset_state::failed;
                    current->second.snapshot.revision = ++revision;
                    current->second.snapshot.diagnostics.push_back(
                        diagnostic(guid, asset_diagnostic_severity::error, "io", read.error().message));
                    persist_record(current->second);
                    emit(asset_event_kind::failed, guid, asset_state::failed, read.error().message);
                }
                return;
            }

            asset_importer* importer{};
            {
                std::shared_lock state_lock(mutex);
                const auto importer_iterator = importers.find(importer_id);
                if (importer_iterator != importers.end()) importer = importer_iterator->second.get();
            }
            if (!importer) return;
            const auto importer_descriptor = importer->descriptor();

            const auto& bytes = read.value();
            asset_import_result imported = importer->import({.reference = {guid, metadata.type, source_path_hint},
                                                             .metadata = metadata,
                                                             .project_root = config.project_root,
                                                             .asset_root = config.asset_root,
                                                             .source_path = source_path,
                                                             .derived_data_root = config.cache_root / "derived",
                                                             .source_bytes = bytes,
                                                             .source_hash = source_hash,
                                                             .priority = priority,
                                                             .requested_residency = import_residency,
                                                             .cancellation = effective_cancellation});

            if (imported.succeeded() && !imported.dependencies.empty())
            {
                std::vector<asset_guid> prerequisite_guids;
                bool invalid_prerequisite{};
                {
                    std::shared_lock dependency_lock(mutex);
                    for (const auto& reference : imported.dependencies)
                    {
                        asset_guid dependency_guid = reference.guid;
                        if (!dependency_guid.valid() && !reference.path_hint.empty())
                            if (const auto path = paths.find(path_key(reference.path_hint)); path != paths.end())
                                dependency_guid = path->second;
                        const auto dependency_record = records.find(dependency_guid);
                        std::unordered_set<asset_guid, asset_guid_hash> visited;
                        if (!dependency_guid.valid() || dependency_record == records.end() ||
                            (reference.expected_type.valid() &&
                             reference.expected_type != dependency_record->second.snapshot.type) ||
                            dependency_reaches(dependency_guid, guid, visited))
                        {
                            invalid_prerequisite = true;
                            break;
                        }
                        if (std::find(prerequisite_guids.begin(), prerequisite_guids.end(), dependency_guid) ==
                            prerequisite_guids.end())
                            prerequisite_guids.push_back(dependency_guid);
                    }
                }
                for (const auto dependency_guid : prerequisite_guids)
                {
                    if (invalid_prerequisite || effective_cancellation.stop_requested()) break;
                    const auto dependency_job = ensure_import(dependency_guid, priority, {});
                    if (dependency_job.valid() && !dependency_job.wait_result().succeeded())
                        invalid_prerequisite = true;
                }
                if (invalid_prerequisite)
                    imported.error = {.code = asset_error_code::dependency_failed,
                                      .guid = guid,
                                      .path = source_path,
                                      .message = "An imported dependency is missing, cyclic, mismatched, or failed"};
            }

            std::unique_lock state_lock(mutex);
            const auto current = records.find(guid);
            if (current == records.end()) return;
            record& target = current->second;
            if (!imported.succeeded())
            {
                target.snapshot.state = asset_state::failed;
                target.snapshot.revision = ++revision;
                target.snapshot.diagnostics.insert(target.snapshot.diagnostics.end(), imported.diagnostics.begin(),
                                                   imported.diagnostics.end());
                target.snapshot.diagnostics.push_back(
                    diagnostic(guid, asset_diagnostic_severity::error, "import",
                               imported.error.message.empty() ? "Asset import failed" : imported.error.message));
                persist_record(target);
                emit(asset_event_kind::failed, guid, asset_state::failed,
                     "Asset import failed; retaining last-good generation");
                return;
            }

            std::vector<asset_guid> resolved_dependencies;
            resolved_dependencies.reserve(imported.dependencies.size());
            bool invalid_dependency{};
            for (const auto& dependency_reference : imported.dependencies)
            {
                asset_guid dependency_guid = dependency_reference.guid;
                if (!dependency_guid.valid() && !dependency_reference.path_hint.empty())
                {
                    if (const auto path = paths.find(path_key(dependency_reference.path_hint)); path != paths.end())
                        dependency_guid = path->second;
                }
                const auto dependency_record = records.find(dependency_guid);
                std::unordered_set<asset_guid, asset_guid_hash> visited;
                if (!dependency_guid.valid() || dependency_record == records.end() ||
                    (dependency_reference.expected_type.valid() &&
                     dependency_reference.expected_type != dependency_record->second.snapshot.type) ||
                    dependency_reaches(dependency_guid, guid, visited))
                {
                    invalid_dependency = true;
                    break;
                }
                if (std::find(resolved_dependencies.begin(), resolved_dependencies.end(), dependency_guid) ==
                    resolved_dependencies.end())
                    resolved_dependencies.push_back(dependency_guid);
            }
            if (invalid_dependency)
            {
                target.snapshot.state = asset_state::failed;
                target.snapshot.revision = ++revision;
                target.snapshot.diagnostics.push_back(
                    diagnostic(guid, asset_diagnostic_severity::error, "dependency",
                               "Importer returned a missing, mismatched, or cyclic dependency"));
                persist_record(target);
                emit(asset_event_kind::failed, guid, asset_state::failed,
                     "Imported dependency validation failed; retaining last-good generation");
                return;
            }
            if (imported.dependencies_authoritative)
            {
                for (const auto old : target.snapshot.dependencies)
                    if (const auto dependency = records.find(old); dependency != records.end())
                        std::erase(dependency->second.snapshot.reverse_dependencies, guid);
                target.snapshot.dependencies = resolved_dependencies;
                for (const auto dependency_guid : resolved_dependencies)
                {
                    auto& reverse = records.at(dependency_guid).snapshot.reverse_dependencies;
                    if (std::find(reverse.begin(), reverse.end(), guid) == reverse.end()) reverse.push_back(guid);
                }
            }

            if (!imported.subassets.empty())
            {
                std::vector<asset_subasset_metadata> merged;
                merged.reserve(imported.subassets.size() + target.metadata.subassets.size());
                for (auto subasset : imported.subassets)
                {
                    const auto old = std::find_if(target.metadata.subassets.begin(), target.metadata.subassets.end(),
                                                  [&](const auto& value)
                                                  { return value.persistent_key == subasset.persistent_key; });
                    if (old != target.metadata.subassets.end())
                        subasset.guid = old->guid;
                    else if (!subasset.guid.valid())
                        subasset.guid = generate_asset_guid();
                    subasset.tombstoned = false;
                    merged.push_back(std::move(subasset));
                }
                for (auto old : target.metadata.subassets)
                {
                    if (std::none_of(merged.begin(), merged.end(),
                                     [&](const auto& value) { return value.persistent_key == old.persistent_key; }))
                    {
                        old.tombstoned = true;
                        merged.push_back(std::move(old));
                    }
                }
                target.metadata.subassets = std::move(merged);
                target.snapshot.subassets = target.metadata.subassets;
                if (!target.snapshot.read_only)
                {
                    auto saved = save_asset_metadata(metadata_path_for(target.absolute_path), target.metadata);
                    if (!saved)
                        target.snapshot.diagnostics.push_back(
                            diagnostic(guid, asset_diagnostic_severity::warning, "metadata", saved.error().message));
                }
            }

            std::vector<asset_hash> dependency_hashes;
            dependency_hashes.reserve(target.snapshot.dependencies.size());
            for (const asset_guid dependency : target.snapshot.dependencies)
            {
                if (const auto dependency_record = records.find(dependency); dependency_record != records.end())
                    dependency_hashes.push_back(!dependency_record->second.snapshot.dependency_hash.empty()
                                                    ? dependency_record->second.snapshot.dependency_hash
                                                    : dependency_record->second.snapshot.source_hash);
            }
            const auto direct_dependencies_hash = combine_hashes(dependency_hashes);
            const auto importer_text = to_string(importer_descriptor.id);
            const auto importer_version_text = std::to_string(importer_descriptor.version);
            std::vector<asset_hash> key_parts{
                target.snapshot.source_hash,
                hash_bytes(
                    std::as_bytes(std::span(metadata.canonical_settings.data(), metadata.canonical_settings.size()))),
                hash_bytes(std::as_bytes(std::span(config.target_profile.data(), config.target_profile.size()))),
                hash_bytes(std::as_bytes(std::span(importer_text.data(), importer_text.size()))),
                hash_bytes(std::as_bytes(std::span(importer_version_text.data(), importer_version_text.size()))),
                direct_dependencies_hash};
            const auto derived_key = combine_hashes(key_parts);
            target.snapshot.dependency_hash = derived_key;
            const auto key_text = to_string(derived_key);
            const auto directory =
                config.cache_root / "derived" / config.target_profile / key_text.substr(0, 2) / key_text;
            std::error_code filesystem_error;
            std::filesystem::create_directories(directory, filesystem_error);
            target.snapshot.artifacts.clear();
            if (!filesystem_error)
            {
                for (const auto& artifact : imported.artifacts)
                {
                    const auto artifact_hash = hash_bytes(artifact.bytes);
                    const auto destination = directory / (artifact.name + artifact.extension);
                    std::string publish_error;
                    if (publish_artifact(destination, artifact.bytes, artifact_hash, publish_error))
                    {
                        target.snapshot.artifacts.push_back({.name = artifact.name,
                                                             .path = destination,
                                                             .content_hash = artifact_hash,
                                                             .size = artifact.bytes.size(),
                                                             .residency = artifact.residency});
                    }
                    else
                        target.snapshot.diagnostics.push_back(diagnostic(guid, asset_diagnostic_severity::error,
                                                                         "derived-data", std::move(publish_error)));
                }
            }

            target.slot->payload.store(std::make_shared<const asset_payload>(std::move(imported.payload)),
                                       std::memory_order_release);
            target.slot->generation.fetch_add(1, std::memory_order_acq_rel);
            target.snapshot.generation = target.slot->generation.load(std::memory_order_acquire);
            target.snapshot.state = asset_state::ready;
            target.snapshot.residency = imported.residency;
            target.snapshot.has_last_good = true;
            target.snapshot.imported_version = importer->descriptor().version;
            target.snapshot.importer_version = importer->descriptor().version;
            target.snapshot.revision = ++revision;
            target.snapshot.diagnostics.insert(target.snapshot.diagnostics.end(), imported.diagnostics.begin(),
                                               imported.diagnostics.end());
            target.last_used = std::chrono::steady_clock::now();
            persist_record(target);
            persist_dependencies(target);
            persist_artifacts(target);
            if (database)
            {
                sqlite_statement generation(
                    database, "INSERT OR REPLACE INTO import_generations("
                              "asset_guid,generation,dependency_hash,published,status) VALUES(?,?,?,?,?);");
                bind_text(generation.get(), 1, to_string(guid));
                sqlite3_bind_int64(generation.get(), 2, static_cast<sqlite3_int64>(target.snapshot.generation));
                bind_text(generation.get(), 3, to_string(target.snapshot.dependency_hash));
                sqlite3_bind_int64(
                    generation.get(), 4,
                    static_cast<sqlite3_int64>(std::chrono::system_clock::now().time_since_epoch().count()));
                sqlite3_bind_int(generation.get(), 5, static_cast<int>(asset_state::ready));
                sqlite3_step(generation.get());
            }
            emit(asset_event_kind::published, guid, asset_state::ready, "Asset generation published", 1.0f);
        });
    return value.active_import;
}

} // namespace arc::assets
