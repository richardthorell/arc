#include <arc/assets/asset_database.h>

namespace arc::assets
{
namespace
{

asset_database_provider provider_for(const asset_snapshot& snapshot)
{
    asset_database_provider provider;
    provider.generation = snapshot.generation;
    provider.active = true;
    provider.read_only = snapshot.read_only;

    const auto source = normalize_asset_path(snapshot.source_path);
    if (source.starts_with("arc://builtin/"))
    {
        provider.id = "arc.builtin.virtual";
        provider.kind = asset_provider_kind::virtual_builtin;
        provider.debug_name = "Engine Built-in";
    }
    else if (snapshot.read_only)
    {
        provider.id = "arc.builtin.source";
        provider.kind = asset_provider_kind::builtin_source;
        provider.debug_name = "Built-in Source";
    }
    else
    {
        provider.id = "arc.project.source";
        provider.kind = asset_provider_kind::project_source;
        provider.debug_name = "Project Source";
    }
    return provider;
}

asset_database_record database_record_from(const asset_snapshot& snapshot)
{
    asset_database_record record;
    record.guid = snapshot.guid;
    record.type = snapshot.type;
    record.title = snapshot.title;
    record.description = snapshot.description;
    record.source_hint = normalize_asset_path(snapshot.source_path);
    record.dependencies = snapshot.dependencies;
    record.reverse_dependencies = snapshot.reverse_dependencies;
    record.state = snapshot.state;
    record.residency = snapshot.residency;
    record.generation = snapshot.generation;
    record.revision = snapshot.revision;
    record.strong_references = snapshot.strong_references;
    record.pins = snapshot.pins;
    record.has_last_good = snapshot.has_last_good;

    auto provider = provider_for(snapshot);
    record.active_provider = provider.id;
    record.providers.push_back(provider);

    record.representations.reserve(snapshot.artifacts.size());
    for (const auto& artifact : snapshot.artifacts)
        record.representations.push_back({.name = artifact.name,
                                          .provider_id = provider.id,
                                          .content_hash = artifact.content_hash,
                                          .size = artifact.size,
                                          .residency = artifact.residency,
                                          .locator_hint = normalize_asset_path(artifact.path)});
    return record;
}

} // namespace

std::optional<asset_database_record> asset_manager_database::query(asset_guid guid, asset_type_id expected_type) const
{
    if (!manager_ || !guid.valid()) return std::nullopt;
    const auto snapshot = manager_->find(guid);
    if (!snapshot || (expected_type.valid() && snapshot->type != expected_type)) return std::nullopt;
    return database_record_from(*snapshot);
}

std::vector<asset_guid> asset_manager_database::dependencies(asset_guid guid) const
{
    return manager_ ? manager_->dependencies(guid) : std::vector<asset_guid>{};
}

std::vector<asset_guid> asset_manager_database::reverse_dependencies(asset_guid guid) const
{
    return manager_ ? manager_->reverse_dependencies(guid) : std::vector<asset_guid>{};
}

std::uint64_t asset_manager_database::revision() const
{
    return manager_ ? manager_->snapshot().revision : 0;
}

} // namespace arc::assets
