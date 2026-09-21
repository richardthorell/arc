#include "asset_manager_internal.h"

#include <algorithm>
#include <mutex>
#include <unordered_set>

namespace arc::assets
{
using namespace manager_detail;

asset_scan_result asset_manager::scan()
{
    asset_scan_result result;
    std::error_code error;
    std::filesystem::create_directories(implementation_->config.asset_root, error);
    if (error)
    {
        result.error = {.code = asset_error_code::io_failed,
                        .path = implementation_->config.asset_root,
                        .message = "Could not create or access the project asset root"};
        return result;
    }

    struct discovered_source
    {
        std::filesystem::path absolute;
        std::filesystem::path relative;
        asset_source_metadata metadata;
        std::filesystem::file_time_type modified;
        std::uint64_t size{};
        asset_hash hash{};
        bool metadata_created{};
        bool hash_deferred{};
        bool read_only{};
    };
    struct mounted_source_root
    {
        std::filesystem::path path;
        bool read_only{};
    };
    std::vector<discovered_source> discovered;
    std::vector<mounted_source_root> source_roots{{implementation_->config.asset_root, false}};
    const auto append_root = [&](const std::filesystem::path& root, bool read_only)
    {
        if (std::none_of(source_roots.begin(), source_roots.end(),
                         [&](const auto& value) { return path_key(value.path) == path_key(root); }))
            source_roots.push_back({root, read_only});
    };
    for (const auto& root : implementation_->config.additional_source_roots)
        append_root(root, false);
    for (const auto& root : implementation_->config.read_only_source_roots)
        append_root(root, true);
    for (const auto& mounted_root : source_roots)
    {
        const auto& source_root = mounted_root.path;
        if (!std::filesystem::exists(source_root, error))
        {
            error.clear();
            continue;
        }
        for (std::filesystem::recursive_directory_iterator
                 iterator(source_root, std::filesystem::directory_options::skip_permission_denied, error),
             end;
             iterator != end; iterator.increment(error))
        {
            if (error)
            {
                error.clear();
                continue;
            }
            if (!iterator->is_regular_file(error)) continue;
            const auto classification = classify_asset_path(iterator->path());
            if (!classification) continue;
            const auto relative =
                mounted_root.read_only
                    ? std::filesystem::path("builtin") / iterator->path().lexically_relative(source_root)
                    : iterator->path().lexically_relative(implementation_->config.project_root);
            if (relative.empty() || relative.native().starts_with(std::filesystem::path("..").native())) continue;

            discovered_source source;
            source.absolute = iterator->path();
            source.relative = relative;
            source.read_only = mounted_root.read_only;
            source.modified = iterator->last_write_time(error);
            source.size = iterator->file_size(error);
            const auto metadata_path = metadata_path_for(source.absolute);
            auto loaded_metadata = load_asset_metadata(metadata_path);
            if (!loaded_metadata)
            {
                if (std::filesystem::exists(metadata_path))
                {
                    result.diagnostics.push_back(
                        {.severity = asset_diagnostic_severity::error,
                         .category = "metadata",
                         .message = normalize_asset_path(relative) + ": " + loaded_metadata.error().message});
                    continue;
                }
                if (mounted_root.read_only)
                {
                    result.diagnostics.push_back(
                        {.severity = asset_diagnostic_severity::error,
                         .category = "metadata",
                         .message = normalize_asset_path(relative) +
                                    ": built-in assets require a checked-in .arcmeta sidecar"});
                    continue;
                }
                if (!implementation_->config.create_missing_metadata) continue;
                source.metadata.guid =
                    authored_asset_guid(source.absolute, classification->first).value_or(generate_asset_guid());
                source.metadata.type = classification->first;
                source.metadata.importer = classification->second;
                auto saved = save_asset_metadata(metadata_path, source.metadata);
                if (!saved)
                {
                    result.diagnostics.push_back(
                        {.severity = asset_diagnostic_severity::error,
                         .category = "metadata",
                         .message = normalize_asset_path(relative) + ": " + saved.error().message});
                    continue;
                }
                source.metadata_created = true;
                ++result.metadata_created;
            }
            else
                source.metadata = std::move(loaded_metadata).value();
            bool hash_source = true;
            if (!source.metadata_created && implementation_->config.enable_source_monitor)
            {
                std::shared_lock state_lock(implementation_->mutex);
                if (const auto existing = implementation_->records.find(source.metadata.guid);
                    existing != implementation_->records.end())
                {
                    if (existing->second.modified == source.modified && existing->second.file_size == source.size)
                    {
                        source.hash = existing->second.snapshot.source_hash;
                        hash_source = false;
                    }
                    else if (implementation_->config.change_debounce.count() > 0)
                    {
                        const auto now = std::chrono::steady_clock::now();
                        const bool settled =
                            existing->second.pending_source_change &&
                            existing->second.pending_modified == source.modified &&
                            existing->second.pending_file_size == source.size &&
                            now - existing->second.pending_since >= implementation_->config.change_debounce;
                        if (!settled)
                        {
                            source.hash_deferred = true;
                            hash_source = false;
                        }
                    }
                }
            }
            std::string hash_error;
            if (hash_source)
            {
                auto hashed = hash_file(source.absolute);
                if (hashed)
                    source.hash = std::move(hashed).value();
                else
                    hash_error = hashed.error().message;
            }
            if (!source.hash_deferred && source.hash.empty())
            {
                result.diagnostics.push_back({.severity = asset_diagnostic_severity::error,
                                              .guid = source.metadata.guid,
                                              .category = "hash",
                                              .message = normalize_asset_path(relative) + ": " + hash_error});
                continue;
            }
            discovered.push_back(std::move(source));
        }
    }

    std::unique_lock lock(implementation_->mutex);
    std::vector<asset_guid> hot_reload;
    std::unordered_set<asset_guid, asset_guid_hash> seen;
    for (auto& source : discovered)
    {
        if (source.metadata_created)
        {
            std::vector<asset_guid> move_candidates;
            for (const auto& [candidate_guid, candidate] : implementation_->records)
            {
                std::error_code exists_error;
                if (!candidate.virtual_asset && candidate.snapshot.type == source.metadata.type &&
                    candidate.snapshot.source_hash == source.hash &&
                    !std::filesystem::exists(candidate.absolute_path, exists_error))
                    move_candidates.push_back(candidate_guid);
            }
            if (move_candidates.size() == 1)
            {
                source.metadata.guid = move_candidates.front();
                auto saved = save_asset_metadata(metadata_path_for(source.absolute), source.metadata);
                if (!saved)
                    result.diagnostics.push_back(implementation_->diagnostic(
                        source.metadata.guid, asset_diagnostic_severity::warning, "move-reconciliation",
                        "Matched a source-only move but could not preserve its sidecar: " + saved.error().message));
            }
            else if (move_candidates.size() > 1)
                result.diagnostics.push_back(implementation_->diagnostic(
                    source.metadata.guid, asset_diagnostic_severity::warning, "move-reconciliation",
                    "Source-only move matches multiple deleted paths; identity requires manual repair"));
        }
        bool duplicate_identity = !seen.insert(source.metadata.guid).second;
        for (const auto& subasset : source.metadata.subassets)
            duplicate_identity = !seen.insert(subasset.guid).second || duplicate_identity;
        if (duplicate_identity)
        {
            result.diagnostics.push_back(implementation_->diagnostic(
                source.metadata.guid, asset_diagnostic_severity::error, "metadata",
                "Duplicate asset GUID encountered at " + normalize_asset_path(source.relative)));
            continue;
        }
        if (source.hash_deferred)
        {
            if (const auto pending = implementation_->records.find(source.metadata.guid);
                pending != implementation_->records.end())
            {
                if (!pending->second.pending_source_change || pending->second.pending_modified != source.modified ||
                    pending->second.pending_file_size != source.size)
                {
                    pending->second.pending_modified = source.modified;
                    pending->second.pending_file_size = source.size;
                    pending->second.pending_since = std::chrono::steady_clock::now();
                    pending->second.pending_source_change = true;
                }
            }
            continue;
        }
        const std::string relative_text = normalize_asset_path(source.relative);
        auto found = implementation_->records.find(source.metadata.guid);
        if (found == implementation_->records.end())
        {
            implementation::record value;
            value.metadata = source.metadata;
            value.absolute_path = source.absolute;
            value.modified = source.modified;
            value.file_size = source.size;
            value.snapshot.guid = source.metadata.guid;
            value.snapshot.type = source.metadata.type;
            value.snapshot.importer = source.metadata.importer;
            value.snapshot.source_path = relative_text;
            value.snapshot.title = source.metadata.title;
            value.snapshot.description = source.metadata.description;
            value.snapshot.source_hash = source.hash;
            value.snapshot.state = asset_state::stale;
            value.snapshot.residency = asset_residency::source;
            value.snapshot.read_only = source.read_only;
            value.snapshot.revision = ++implementation_->revision;
            value.snapshot.subassets = source.metadata.subassets;
            value.slot->requested_guid = value.snapshot.guid;
            value.slot->resolved_guid = value.snapshot.guid;
            value.slot->type = value.snapshot.type;
            if (const auto importer = implementation_->importers.find(value.snapshot.importer);
                importer != implementation_->importers.end())
                value.snapshot.importer_version = importer->second->descriptor().version;
            implementation_->paths[path_key(relative_text)] = value.snapshot.guid;
            auto [inserted, _] = implementation_->records.emplace(value.snapshot.guid, std::move(value));
            implementation_->persist_record(inserted->second);
            implementation_->clear_tombstone(source.metadata.guid);
            implementation_->emit(asset_event_kind::discovered, source.metadata.guid, asset_state::stale,
                                  "Asset discovered");
            ++result.discovered;
            continue;
        }

        implementation::record& value = found->second;
        const bool observed_file_change = value.modified != source.modified || value.file_size != source.size;
        bool debounce_complete = true;
        if (observed_file_change && implementation_->config.enable_source_monitor &&
            implementation_->config.change_debounce.count() > 0)
        {
            const auto now = std::chrono::steady_clock::now();
            if (!value.pending_source_change || value.pending_modified != source.modified ||
                value.pending_file_size != source.size)
            {
                value.pending_modified = source.modified;
                value.pending_file_size = source.size;
                value.pending_since = now;
                value.pending_source_change = true;
                debounce_complete = false;
            }
            else
                debounce_complete = now - value.pending_since >= implementation_->config.change_debounce;
        }
        const bool source_changed = debounce_complete && value.snapshot.source_hash != source.hash;
        const bool settings_changed = value.metadata.importer != source.metadata.importer ||
                                      value.metadata.settings_version != source.metadata.settings_version ||
                                      value.metadata.canonical_settings != source.metadata.canonical_settings;
        std::uint32_t registered_importer_version{};
        if (const auto importer = implementation_->importers.find(source.metadata.importer);
            importer != implementation_->importers.end())
            registered_importer_version = importer->second->descriptor().version;
        const bool importer_changed = registered_importer_version != 0 && value.snapshot.imported_version != 0 &&
                                      value.snapshot.imported_version != registered_importer_version;
        const bool path_changed = normalize_asset_path(value.snapshot.source_path) != relative_text;
        if (path_changed)
        {
            implementation_->paths.erase(path_key(value.snapshot.source_path));
            implementation_->paths[path_key(relative_text)] = value.snapshot.guid;
            value.snapshot.source_path = relative_text;
            implementation_->emit(asset_event_kind::moved, value.snapshot.guid, value.snapshot.state,
                                  "Asset path changed");
        }
        value.snapshot.title = source.metadata.title;
        value.snapshot.description = source.metadata.description;
        value.metadata = source.metadata;
        value.absolute_path = source.absolute;
        if (!observed_file_change || debounce_complete)
        {
            value.modified = source.modified;
            value.file_size = source.size;
            value.pending_source_change = false;
        }
        value.snapshot.source_missing = false;
        value.snapshot.read_only = source.read_only;
        value.snapshot.type = source.metadata.type;
        value.snapshot.importer = source.metadata.importer;
        value.snapshot.importer_version = registered_importer_version;
        value.snapshot.subassets = source.metadata.subassets;
        if (source_changed || settings_changed || importer_changed)
        {
            value.snapshot.source_hash = source.hash;
            value.snapshot.state = asset_state::stale;
            value.snapshot.revision = ++implementation_->revision;
            implementation_->mark_reverse_stale(value.snapshot.guid, "source or import settings changed");
            implementation_->emit(asset_event_kind::state_changed, value.snapshot.guid, asset_state::stale,
                                  "Asset source changed");
            if (implementation_->config.enable_source_monitor && value.snapshot.has_last_good)
                hot_reload.push_back(value.snapshot.guid);
            ++result.updated;
        }
        else if (path_changed)
        {
            value.snapshot.revision = ++implementation_->revision;
            ++result.updated;
        }
        implementation_->persist_record(value);
        implementation_->clear_tombstone(source.metadata.guid);
    }

    for (auto& [guid, value] : implementation_->records)
    {
        if (value.virtual_asset || seen.contains(guid) || value.snapshot.source_missing) continue;
        value.snapshot.source_missing = true;
        value.snapshot.state = asset_state::unknown;
        value.snapshot.revision = ++implementation_->revision;
        value.snapshot.diagnostics.push_back(implementation_->diagnostic(guid, asset_diagnostic_severity::error,
                                                                         "missing", "Asset source file is missing"));
        implementation_->persist_record(value);
        implementation_->persist_tombstone(value);
        implementation_->emit(asset_event_kind::failed, guid, asset_state::unknown, "Asset source file is missing");
        ++result.missing;
    }
    if (discovered.empty() && error)
    {
        result.diagnostics.push_back(
            implementation_->diagnostic({}, asset_diagnostic_severity::warning, "scan", error.message()));
    }
    lock.unlock();
    for (const auto guid : hot_reload)
        implementation_->ensure_import(guid, asset_streaming_priority::high, {});
    return result;
}

} // namespace arc::assets
