#include "asset_manager_internal.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <mutex>
#include <thread>

namespace arc::assets
{
namespace manager_detail
{

constexpr std::uint32_t registry_schema_version = 2;

std::string path_key(std::string value)
{
#if defined(_WIN32)
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
#endif
    return value;
}

std::string path_key(const std::filesystem::path& value)
{
    return path_key(normalize_asset_path(value));
}

bool is_shader_include_path(const std::filesystem::path& path)
{
    const auto extension = path.extension().string();
    if (extension == ".inc") return true;
    if (extension != ".slang" && extension != ".glsl" && extension != ".hlsl") return false;
    return std::any_of(path.begin(), path.end(),
                       [](const auto& component) { return component == std::filesystem::path("include"); });
}

bool path_within(const std::filesystem::path& root, const std::filesystem::path& candidate)
{
    const auto normalized_root = std::filesystem::absolute(root).lexically_normal();
    const auto normalized_candidate = std::filesystem::absolute(candidate).lexically_normal();
    auto root_iterator = normalized_root.begin();
    auto candidate_iterator = normalized_candidate.begin();
    for (; root_iterator != normalized_root.end(); ++root_iterator, ++candidate_iterator)
        if (candidate_iterator == normalized_candidate.end() ||
            path_key(*root_iterator) != path_key(*candidate_iterator))
            return false;
    return true;
}

std::optional<asset_guid> authored_asset_guid(const std::filesystem::path& source, asset_type_id type)
{
    if (type != asset_types::prefab) return std::nullopt;
    std::ifstream stream(source, std::ios::binary);
    if (!stream) return std::nullopt;
    const auto document = nlohmann::json::parse(stream, nullptr, false);
    if (!document.is_object() || document.value("format", "") != "arc.prefab" || !document.contains("prefab") ||
        !document["prefab"].is_object() || !document["prefab"].contains("id") || !document["prefab"]["id"].is_string())
        return std::nullopt;
    return parse_asset_guid(document["prefab"]["id"].get<std::string>());
}

bool publish_artifact(const std::filesystem::path& destination, std::span<const std::byte> bytes,
                      const asset_hash& expected_hash, std::string& error)
{
    const auto matches_hash = [&](const std::filesystem::path& path)
    {
        auto hashed = hash_file(path);
        if (!hashed)
        {
            error = hashed.error().message;
            return false;
        }
        return hashed.value() == expected_hash;
    };
    std::error_code filesystem_error;
    if (std::filesystem::exists(destination, filesystem_error) && !filesystem_error)
    {
        const auto existing_size = std::filesystem::file_size(destination, filesystem_error);
        if (!filesystem_error && existing_size == bytes.size() && matches_hash(destination)) return true;
        error = "derived artifact already exists with unexpected contents";
        return false;
    }

    const auto temporary = std::filesystem::path(
        destination.string() + ".tmp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
        "-" + std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            error = "could not create temporary derived artifact";
            return false;
        }
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        output.flush();
        if (!output)
        {
            error = "failed while writing derived artifact";
            return false;
        }
    }
    if (!matches_hash(temporary))
    {
        std::filesystem::remove(temporary, filesystem_error);
        error = "derived artifact failed hash verification";
        return false;
    }
    std::filesystem::rename(temporary, destination, filesystem_error);
    if (filesystem_error)
    {
        filesystem_error.clear();
        if (std::filesystem::exists(destination, filesystem_error) && !filesystem_error &&
            std::filesystem::file_size(destination, filesystem_error) == bytes.size() && !filesystem_error &&
            matches_hash(destination))
        {
            std::filesystem::remove(temporary, filesystem_error);
            return true;
        }
        std::filesystem::remove(temporary, filesystem_error);
        error = "could not publish derived artifact";
        return false;
    }
    return true;
}

std::int64_t file_time_value(const std::filesystem::file_time_type& value) noexcept
{
    return static_cast<std::int64_t>(value.time_since_epoch().count());
}

sqlite_statement::sqlite_statement(sqlite3* database, const char* sql)
{
    sqlite3_prepare_v2(database, sql, -1, &statement_, nullptr);
}

sqlite_statement::~sqlite_statement()
{
    if (statement_) sqlite3_finalize(statement_);
}

sqlite3_stmt* sqlite_statement::get() const noexcept
{
    return statement_;
}

sqlite_statement::operator bool() const noexcept
{
    return statement_ != nullptr;
}

bool execute(sqlite3* database, const char* sql, std::string* error = nullptr)
{
    char* sqlite_error{};
    const int result = sqlite3_exec(database, sql, nullptr, nullptr, &sqlite_error);
    if (result == SQLITE_OK) return true;
    if (error) *error = sqlite_error ? sqlite_error : sqlite3_errmsg(database);
    sqlite3_free(sqlite_error);
    return false;
}

void bind_text(sqlite3_stmt* statement, int index, std::string_view value)
{
    // SQLITE_TRANSIENT is SQLite's documented sentinel for copying bound data.
    sqlite3_bind_text(statement, index, value.data(), static_cast<int>(value.size()),
                      SQLITE_TRANSIENT); // NOLINT(performance-no-int-to-ptr)
}

std::string column_text(sqlite3_stmt* statement, int column)
{
    const auto* value = sqlite3_column_text(statement, column);
    return value ? reinterpret_cast<const char*>(value) : std::string{};
}

asset_reference dependency_from_path(const asset_import_context& context, std::string_view text,
                                     asset_type_id expected_type = {})
{
    if (text.empty() || text.starts_with("data:") || text.find("://") != std::string_view::npos) return {};
    std::filesystem::path authored(text);
    if (authored.is_absolute() || authored.has_root_name()) return {};

    const auto normalized_text = normalize_asset_path(authored);
    const auto source_hint = normalize_asset_path(context.reference.path_hint);
    if (source_hint == "builtin" || source_hint.starts_with("builtin/"))
    {
        const auto mounted_relative = std::filesystem::path(source_hint).lexically_relative("builtin");
        auto mounted_root = context.source_path;
        for (const auto& component : mounted_relative)
        {
            (void)component;
            mounted_root = mounted_root.parent_path();
        }
        const auto resolved =
            context.metadata.type == asset_types::material || context.metadata.type == asset_types::material_instance
                ? mounted_root / authored
                : context.source_path.parent_path() / authored;
        const auto relative_to_mount = resolved.lexically_normal().lexically_relative(mounted_root);
        if (relative_to_mount.empty() || relative_to_mount.native().starts_with(std::filesystem::path("..").native()))
            return {};
        const auto path_hint = std::filesystem::path("builtin") / relative_to_mount;
        if (!expected_type.valid())
            if (const auto classification = classify_asset_path(path_hint)) expected_type = classification->first;
        return {.expected_type = expected_type, .path_hint = normalize_asset_path(path_hint)};
    }

    const auto configured_asset_root =
        context.asset_root.empty() ? context.project_root / "assets" : context.asset_root;
    const auto relative_asset_root =
        normalize_asset_path(configured_asset_root.lexically_normal().lexically_relative(context.project_root));
    const auto normalized_key = path_key(normalized_text);
    const auto asset_root_key = path_key(relative_asset_root);
    const bool already_rooted = !asset_root_key.empty() &&
                                (normalized_key == asset_root_key || normalized_key.starts_with(asset_root_key + "/"));

    std::filesystem::path resolved;
    if (already_rooted)
        resolved = context.project_root / authored;
    else if (context.metadata.type == asset_types::material || context.metadata.type == asset_types::material_instance)
        resolved = configured_asset_root / authored;
    else
        resolved = context.source_path.parent_path() / authored;
    const auto relative = resolved.lexically_normal().lexically_relative(context.project_root);
    if (relative.empty() || relative.native().starts_with(std::filesystem::path("..").native())) return {};
    if (!expected_type.valid())
        if (const auto classification = classify_asset_path(relative)) expected_type = classification->first;
    return {.expected_type = expected_type, .path_hint = normalize_asset_path(relative)};
}

void append_dependency(std::vector<asset_reference>& output, asset_reference reference)
{
    if (!reference.guid.valid() && reference.path_hint.empty()) return;
    if (std::none_of(output.begin(), output.end(),
                     [&](const auto& value)
                     {
                         return reference.guid.valid() ? value.guid == reference.guid
                                                       : normalize_asset_path(value.path_hint) ==
                                                             normalize_asset_path(reference.path_hint);
                     }))
        output.push_back(std::move(reference));
}

void collect_json_dependencies(const nlohmann::json& value, const asset_import_context& context,
                               std::vector<asset_reference>& output, std::string_view parent_key = {})
{
    if (value.is_object())
    {
        if (value.contains("guid") && value["guid"].is_string() && value.contains("pathHint") &&
            value["pathHint"].is_string())
        {
            asset_reference reference;
            reference.guid = parse_asset_guid(value["guid"].get<std::string>()).value_or(asset_guid{});
            if (value.contains("expectedType") && value["expectedType"].is_string())
                reference.expected_type =
                    parse_asset_type_id(value["expectedType"].get<std::string>()).value_or(asset_type_id{});
            const auto path_reference =
                dependency_from_path(context, value["pathHint"].get<std::string>(), reference.expected_type);
            reference.path_hint = path_reference.path_hint;
            if (!reference.expected_type.valid()) reference.expected_type = path_reference.expected_type;
            append_dependency(output, std::move(reference));
        }
        for (const auto& [key, child] : value.items())
            collect_json_dependencies(child, context, output, key);
        return;
    }
    if (value.is_array())
    {
        for (const auto& child : value)
            collect_json_dependencies(child, context, output, parent_key);
        return;
    }
    if (!value.is_string()) return;
    const auto text = value.get<std::string>();
    const bool likely_asset_path = parent_key == "uri" || parent_key == "path" || parent_key == "material" ||
                                   parent_key == "prefabPath" || parent_key == "pathHint" ||
                                   classify_asset_path(std::filesystem::path(text)).has_value();
    if (likely_asset_path) append_dependency(output, dependency_from_path(context, text));
}

void collect_shader_dependencies(const asset_import_context& context, std::vector<asset_reference>& output)
{
    const std::string source(reinterpret_cast<const char*>(context.source_bytes.data()), context.source_bytes.size());
    std::size_t cursor{};
    while ((cursor = source.find("#include", cursor)) != std::string::npos)
    {
        const auto quote = source.find_first_of("\"<", cursor + 8);
        if (quote == std::string::npos) break;
        const char close = source[quote] == '"' ? '"' : '>';
        const auto end = source.find(close, quote + 1);
        if (end == std::string::npos) break;
        append_dependency(output,
                          dependency_from_path(context, std::string_view(source).substr(quote + 1, end - quote - 1),
                                               asset_types::binary_blob));
        cursor = end + 1;
    }
}

class source_blob_importer final : public asset_importer
{
public:
    source_blob_importer(asset_importer_id id, asset_type_id type, std::string name,
                         std::vector<std::string> extensions, std::uint32_t settings_version = 1)
    {
        descriptor_.id = id;
        descriptor_.name = std::move(name);
        descriptor_.extensions = std::move(extensions);
        descriptor_.settings_version = settings_version;
        descriptor_.output_types.push_back(type);
    }

    const asset_importer_descriptor& descriptor() const noexcept override
    {
        return descriptor_;
    }

    asset_import_result import(const asset_import_context& context) override
    {
        if (context.cancellation.stop_requested())
        {
            return {.error = {.code = asset_error_code::cancelled,
                              .guid = context.reference.guid,
                              .path = context.source_path,
                              .message = "Asset import was cancelled"}};
        }
        auto data = std::make_shared<source_asset_data>();
        data->source_path = context.source_path;
        data->source_hash = context.source_hash;
        data->bytes.assign(context.source_bytes.begin(), context.source_bytes.end());
        asset_import_result result;
        result.payload =
            asset_payload::make<source_asset_data>(context.metadata.type, std::move(data), context.source_bytes.size());
        result.artifacts.push_back(
            {.name = "source",
             .extension = ".bin",
             .bytes = std::vector<std::byte>(context.source_bytes.begin(), context.source_bytes.end()),
             .residency = asset_residency::derived});
        if (context.metadata.type == asset_types::scene || context.metadata.type == asset_types::prefab ||
            context.metadata.type == asset_types::material || context.metadata.type == asset_types::material_instance ||
            context.source_path.extension() == ".gltf")
        {
            const auto document = nlohmann::json::parse(reinterpret_cast<const char*>(context.source_bytes.data()),
                                                        reinterpret_cast<const char*>(context.source_bytes.data()) +
                                                            context.source_bytes.size(),
                                                        nullptr, false);
            if (!document.is_discarded()) collect_json_dependencies(document, context, result.dependencies);
        }
        else if (context.metadata.type == asset_types::shader || is_shader_include_path(context.source_path))
            collect_shader_dependencies(context, result.dependencies);
        return result;
    }

private:
    asset_importer_descriptor descriptor_;
};

std::vector<std::unique_ptr<asset_importer>> default_importers()
{
    std::vector<std::unique_ptr<asset_importer>> result;
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::scene, asset_types::scene, "ARC Scene",
                                                            std::vector<std::string>{".arcscene"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::prefab, asset_types::prefab, "ARC Prefab",
                                                            std::vector<std::string>{".arcprefab"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::material, asset_types::material,
                                                            "ARC Material", std::vector<std::string>{".arcmat"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::material_instance,
                                                            asset_types::material_instance, "ARC Material Instance",
                                                            std::vector<std::string>{".arcmatinst"}));
    result.push_back(std::make_unique<source_blob_importer>(
        importer_ids::shader, asset_types::shader, "ARC Shader",
        std::vector<std::string>{".slang", ".glsl", ".vert", ".frag", ".comp", ".hlsl", ".inc"}));
    result.push_back(std::make_unique<source_blob_importer>(
        importer_ids::texture, asset_types::texture_2d, "Texture",
        std::vector<std::string>{".png", ".jpg", ".jpeg", ".dds", ".tga", ".bmp", ".ktx", ".ktx2"}, 2));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::environment, asset_types::environment,
                                                            "Environment", std::vector<std::string>{".hdr", ".exr"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::gltf, asset_types::imported_scene, "glTF",
                                                            std::vector<std::string>{".glb", ".gltf"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::fbx, asset_types::imported_scene, "FBX",
                                                            std::vector<std::string>{".fbx"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::obj, asset_types::imported_scene,
                                                            "Wavefront OBJ", std::vector<std::string>{".obj"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::binary, asset_types::binary_blob,
                                                            "Binary source", std::vector<std::string>{".bin"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::animation, asset_types::animation_clip,
                                                            "Animation", std::vector<std::string>{".arcanim"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::collision, asset_types::collision,
                                                            "Collision", std::vector<std::string>{".arccollision"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::navigation, asset_types::navigation,
                                                            "Navigation", std::vector<std::string>{".arcnav"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::audio, asset_types::audio_clip, "Audio",
                                                            std::vector<std::string>{".wav", ".ogg", ".mp3", ".flac"}));
    return result;
}

} // namespace manager_detail

using namespace manager_detail;

asset_manager::implementation::implementation(asset_manager_config value, jobs::job_system& scheduler,
                                              io::async_file_service& file_service,
                                              memory::memory_system& memory_system)
    : config(std::move(value)), jobs(&scheduler), files(&file_service), memory(&memory_system)
{
#if defined(ARC_BUILD_SHIPPING)
    config.enable_source_monitor = false;
#endif
    if (config.project_root.empty()) config.project_root = std::filesystem::current_path();
    if (config.asset_root.empty()) config.asset_root = config.project_root / "assets";
    if (config.cache_root.empty()) config.cache_root = config.project_root / ".arc" / "cache";
    config.project_root = std::filesystem::absolute(config.project_root).lexically_normal();
    config.asset_root = std::filesystem::absolute(config.asset_root).lexically_normal();
    config.cache_root = std::filesystem::absolute(config.cache_root).lexically_normal();
    for (auto& root : config.additional_source_roots)
    {
        if (root.is_relative()) root = config.project_root / root;
        root = std::filesystem::absolute(root).lexically_normal();
    }
    std::erase_if(config.additional_source_roots,
                  [&](const auto& root) { return !path_within(config.project_root, root); });
    for (auto& root : config.read_only_source_roots)
    {
        if (root.is_relative()) root = config.project_root / root;
        root = std::filesystem::absolute(root).lexically_normal();
    }
    std::erase_if(config.read_only_source_roots,
                  [](const auto& root) { return root.empty() || !std::filesystem::exists(root); });
    streaming = std::make_unique<memory::streaming_heap>(*memory, config.streaming_heap_bytes);
    for (auto& importer : default_importers())
        importers.emplace(importer->descriptor().id, std::move(importer));
}

asset_manager::implementation::~implementation()
{
    if (pressure_handler) memory->remove_pressure_handler(pressure_handler);
    if (database) sqlite3_close(database);
}

bool asset_manager::implementation::managed_source_path(const std::filesystem::path& path) const
{
    return path_within(config.asset_root, path) ||
           std::any_of(config.additional_source_roots.begin(), config.additional_source_roots.end(),
                       [&](const auto& root) { return path_within(root, path); });
}

void asset_manager::implementation::emit(asset_event_kind kind, asset_guid guid, asset_state state, std::string message,
                                         float progress)
{
    asset_event event{.sequence = next_event++,
                      .registry_revision = revision,
                      .kind = kind,
                      .guid = guid,
                      .state = state,
                      .progress = progress,
                      .message = std::move(message)};
    events.push_back(event);
    if (events.size() > 4096) events.erase(events.begin(), events.begin() + 1024);
    std::vector<asset_event_callback> callbacks;
    callbacks.reserve(subscribers.size());
    for (const auto& [_, callback] : subscribers)
        if (callback) callbacks.push_back(callback);
    if (!callbacks.empty() && jobs)
        jobs->dispatch({.name = "assets.event",
                        .priority = jobs::job_priority::low,
                        .affinity = jobs::job_affinity::any_worker,
                        .dependencies = {},
                        .dependency_view = {},
                        .parent = {},
                        .cancellation = {},
                        .dependency_policy = jobs::job_dependency_policy::cancel_on_failure},
                       [callbacks = std::move(callbacks), event]
                       {
                           for (const auto& callback : callbacks)
                               callback(event);
                       });
}

asset_diagnostic asset_manager::implementation::diagnostic(asset_guid guid, asset_diagnostic_severity severity,
                                                           std::string category, std::string message)
{
    return {.sequence = next_diagnostic++,
            .severity = severity,
            .guid = guid,
            .category = std::move(category),
            .message = std::move(message)};
}

bool asset_manager::implementation::open_database(std::string& error)
{
    std::error_code filesystem_error;
    std::filesystem::create_directories(config.cache_root, filesystem_error);
    std::filesystem::create_directories(config.cache_root / "derived", filesystem_error);
    if (filesystem_error)
    {
        error = "Could not create the ARC asset cache directory";
        return false;
    }
    const auto path = config.cache_root / "assets.db";
    if (sqlite3_open_v2(path.string().c_str(), &database,
                        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, nullptr) != SQLITE_OK)
    {
        error = database ? sqlite3_errmsg(database) : "Could not open SQLite asset registry";
        return false;
    }
    execute(database, "PRAGMA journal_mode=WAL;");
    execute(database, "PRAGMA synchronous=NORMAL;");
    execute(database, "PRAGMA foreign_keys=ON;");
    {
        sqlite_statement integrity(database, "PRAGMA quick_check;");
        if (!integrity || sqlite3_step(integrity.get()) != SQLITE_ROW || column_text(integrity.get(), 0) != "ok")
        {
            error = "SQLite asset registry failed its integrity check";
            return false;
        }
    }
    const char* schema =
        "BEGIN;"
        "CREATE TABLE IF NOT EXISTS registry_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
        "CREATE TABLE IF NOT EXISTS assets("
        "guid TEXT PRIMARY KEY,type TEXT NOT NULL,importer TEXT NOT NULL,source_path TEXT NOT NULL,"
        "source_hash TEXT NOT NULL,dependency_hash TEXT NOT NULL,state INTEGER NOT NULL,"
        "residency INTEGER NOT NULL,generation INTEGER NOT NULL,revision INTEGER NOT NULL,"
        "importer_version INTEGER NOT NULL,imported_version INTEGER NOT NULL,"
        "source_missing INTEGER NOT NULL,has_last_good INTEGER NOT NULL,"
        "modified INTEGER NOT NULL,file_size INTEGER NOT NULL);"
        "CREATE UNIQUE INDEX IF NOT EXISTS assets_path ON assets(source_path COLLATE NOCASE);"
        "CREATE TABLE IF NOT EXISTS dependencies("
        "asset_guid TEXT NOT NULL,dependency_guid TEXT NOT NULL,"
        "PRIMARY KEY(asset_guid,dependency_guid));"
        "CREATE INDEX IF NOT EXISTS dependencies_reverse ON dependencies(dependency_guid);"
        "CREATE TABLE IF NOT EXISTS artifacts("
        "asset_guid TEXT NOT NULL,name TEXT NOT NULL,path TEXT NOT NULL,content_hash TEXT NOT NULL,"
        "size INTEGER NOT NULL,residency INTEGER NOT NULL,PRIMARY KEY(asset_guid,name));"
        "CREATE TABLE IF NOT EXISTS subassets("
        "asset_guid TEXT NOT NULL,persistent_key TEXT NOT NULL,guid TEXT NOT NULL,type TEXT NOT NULL,"
        "name TEXT NOT NULL,tombstoned INTEGER NOT NULL,PRIMARY KEY(asset_guid,persistent_key));"
        "CREATE UNIQUE INDEX IF NOT EXISTS subassets_guid ON subassets(guid);"
        "CREATE TABLE IF NOT EXISTS diagnostics("
        "asset_guid TEXT NOT NULL,sequence INTEGER NOT NULL,severity INTEGER NOT NULL,"
        "category TEXT NOT NULL,message TEXT NOT NULL,PRIMARY KEY(asset_guid,sequence));"
        "CREATE TABLE IF NOT EXISTS tombstones("
        "guid TEXT PRIMARY KEY,type TEXT NOT NULL,last_path TEXT NOT NULL,deleted_revision INTEGER NOT NULL);"
        "CREATE TABLE IF NOT EXISTS import_generations("
        "asset_guid TEXT NOT NULL,generation INTEGER NOT NULL,dependency_hash TEXT NOT NULL,"
        "published INTEGER NOT NULL,status INTEGER NOT NULL,PRIMARY KEY(asset_guid,generation));"
        "INSERT OR IGNORE INTO registry_meta(key,value) VALUES('schema_version','1');"
        "COMMIT;";
    if (!execute(database, schema, &error)) return false;
    std::uint32_t current_version{};
    {
        sqlite_statement version(database, "SELECT value FROM registry_meta WHERE key='schema_version';");
        if (!version || sqlite3_step(version.get()) != SQLITE_ROW)
        {
            error = "Asset registry has no schema version";
            return false;
        }
        const auto text = column_text(version.get(), 0);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), current_version);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        {
            error = "Asset registry has an invalid schema version";
            return false;
        }
    }
    if (current_version > registry_schema_version)
    {
        error = "Asset registry schema is newer or incompatible";
        return false;
    }
    if (current_version < registry_schema_version &&
        !execute(database,
                 "BEGIN;"
                 "UPDATE registry_meta SET value='2' WHERE key='schema_version';"
                 "COMMIT;",
                 &error))
        return false;
    return load_database(error);
}

bool asset_manager::implementation::rebuild_database(std::string& error)
{
    if (database)
    {
        sqlite3_close(database);
        database = nullptr;
    }
    records.clear();
    paths.clear();
    const auto database_path = config.cache_root / "assets.db";
    const auto corrupt_path =
        config.cache_root /
        ("assets.db.corrupt-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
    std::error_code filesystem_error;
    if (std::filesystem::exists(database_path, filesystem_error))
    {
        std::filesystem::rename(database_path, corrupt_path, filesystem_error);
        if (filesystem_error)
        {
            error = "Could not preserve the incompatible asset registry: " + filesystem_error.message();
            return false;
        }
    }
    std::filesystem::remove(database_path.string() + "-wal", filesystem_error);
    filesystem_error.clear();
    std::filesystem::remove(database_path.string() + "-shm", filesystem_error);
    return open_database(error);
}

bool asset_manager::implementation::load_database(std::string& error)
{
    sqlite_statement statement(database,
                               "SELECT guid,type,importer,source_path,source_hash,dependency_hash,state,residency,"
                               "generation,revision,importer_version,imported_version,source_missing,has_last_good,"
                               "modified,file_size FROM assets;");
    if (!statement)
    {
        error = sqlite3_errmsg(database);
        return false;
    }
    while (sqlite3_step(statement.get()) == SQLITE_ROW)
    {
        const auto guid = parse_asset_guid(column_text(statement.get(), 0));
        const auto type = parse_asset_type_id(column_text(statement.get(), 1));
        const auto importer = parse_asset_importer_id(column_text(statement.get(), 2));
        const auto source_hash = parse_asset_hash(column_text(statement.get(), 4));
        const auto dependency_hash = parse_asset_hash(column_text(statement.get(), 5));
        if (!guid || !type || !importer) continue;
        record value;
        value.snapshot.guid = *guid;
        value.snapshot.type = *type;
        value.snapshot.importer = *importer;
        value.snapshot.source_path = column_text(statement.get(), 3);
        value.snapshot.source_hash = source_hash.value_or(asset_hash{});
        value.snapshot.dependency_hash = dependency_hash.value_or(asset_hash{});
        value.snapshot.state = static_cast<asset_state>(sqlite3_column_int(statement.get(), 6));
        value.snapshot.residency = static_cast<asset_residency>(sqlite3_column_int(statement.get(), 7));
        value.snapshot.generation = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 8));
        value.snapshot.revision = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 9));
        value.snapshot.importer_version = static_cast<std::uint32_t>(sqlite3_column_int(statement.get(), 10));
        value.snapshot.imported_version = static_cast<std::uint32_t>(sqlite3_column_int(statement.get(), 11));
        value.snapshot.source_missing = sqlite3_column_int(statement.get(), 12) != 0;
        value.snapshot.has_last_good = sqlite3_column_int(statement.get(), 13) != 0;
        value.snapshot.read_only = normalize_asset_path(value.snapshot.source_path).starts_with("builtin/");
        value.modified = std::filesystem::file_time_type(
            std::filesystem::file_time_type::duration(sqlite3_column_int64(statement.get(), 14)));
        value.file_size = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 15));
        value.absolute_path = config.project_root / value.snapshot.source_path;
        value.slot->requested_guid = *guid;
        value.slot->resolved_guid = *guid;
        value.slot->type = *type;
        value.slot->generation = value.snapshot.generation;
        paths[path_key(value.snapshot.source_path)] = *guid;
        records.emplace(*guid, std::move(value));
        revision = std::max(revision, records[*guid].snapshot.revision);
    }

    sqlite_statement dependencies_statement(
        database, "SELECT asset_guid,dependency_guid FROM dependencies ORDER BY asset_guid,dependency_guid;");
    while (dependencies_statement && sqlite3_step(dependencies_statement.get()) == SQLITE_ROW)
    {
        const auto owner = parse_asset_guid(column_text(dependencies_statement.get(), 0));
        const auto dependency = parse_asset_guid(column_text(dependencies_statement.get(), 1));
        if (!owner || !dependency) continue;
        if (const auto found = records.find(*owner); found != records.end())
            found->second.snapshot.dependencies.push_back(*dependency);
        if (const auto found = records.find(*dependency); found != records.end())
            found->second.snapshot.reverse_dependencies.push_back(*owner);
    }
    sqlite_statement artifacts_statement(database,
                                         "SELECT asset_guid,name,path,content_hash,size,residency FROM artifacts "
                                         "ORDER BY asset_guid,name;");
    while (artifacts_statement && sqlite3_step(artifacts_statement.get()) == SQLITE_ROW)
    {
        const auto owner = parse_asset_guid(column_text(artifacts_statement.get(), 0));
        const auto hash = parse_asset_hash(column_text(artifacts_statement.get(), 3));
        if (!owner || !hash) continue;
        if (const auto found = records.find(*owner); found != records.end())
        {
            const std::filesystem::path path = column_text(artifacts_statement.get(), 2);
            const auto size = static_cast<std::uint64_t>(sqlite3_column_int64(artifacts_statement.get(), 4));
            std::error_code artifact_error;
            auto hashed = hash_file(path);
            if (!std::filesystem::exists(path, artifact_error) ||
                std::filesystem::file_size(path, artifact_error) != size || artifact_error || !hashed ||
                hashed.value() != *hash)
            {
                found->second.snapshot.state = asset_state::stale;
                found->second.snapshot.diagnostics.push_back(
                    diagnostic(*owner, asset_diagnostic_severity::warning, "derived-data",
                               "Derived artifact is missing or corrupt and will be regenerated"));
                continue;
            }
            found->second.snapshot.artifacts.push_back(
                {.name = column_text(artifacts_statement.get(), 1),
                 .path = path,
                 .content_hash = *hash,
                 .size = size,
                 .residency = static_cast<asset_residency>(sqlite3_column_int(artifacts_statement.get(), 5))});
        }
    }
    sqlite_statement subassets_statement(database,
                                         "SELECT asset_guid,persistent_key,guid,type,name,tombstoned FROM subassets "
                                         "ORDER BY asset_guid,persistent_key;");
    while (subassets_statement && sqlite3_step(subassets_statement.get()) == SQLITE_ROW)
    {
        const auto owner = parse_asset_guid(column_text(subassets_statement.get(), 0));
        const auto guid = parse_asset_guid(column_text(subassets_statement.get(), 2));
        const auto type = parse_asset_type_id(column_text(subassets_statement.get(), 3));
        if (!owner || !guid || !type) continue;
        if (const auto found = records.find(*owner); found != records.end())
            found->second.snapshot.subassets.push_back(
                {.persistent_key = column_text(subassets_statement.get(), 1),
                 .guid = *guid,
                 .type = *type,
                 .name = column_text(subassets_statement.get(), 4),
                 .tombstoned = sqlite3_column_int(subassets_statement.get(), 5) != 0});
    }
    sqlite_statement diagnostics_statement(database,
                                           "SELECT asset_guid,sequence,severity,category,message FROM diagnostics "
                                           "ORDER BY asset_guid,sequence;");
    while (diagnostics_statement && sqlite3_step(diagnostics_statement.get()) == SQLITE_ROW)
    {
        const auto owner = parse_asset_guid(column_text(diagnostics_statement.get(), 0));
        if (!owner) continue;
        if (const auto found = records.find(*owner); found != records.end())
        {
            const auto sequence = static_cast<std::uint64_t>(sqlite3_column_int64(diagnostics_statement.get(), 1));
            found->second.snapshot.diagnostics.push_back(
                {.sequence = sequence,
                 .severity = static_cast<asset_diagnostic_severity>(sqlite3_column_int(diagnostics_statement.get(), 2)),
                 .guid = *owner,
                 .category = column_text(diagnostics_statement.get(), 3),
                 .message = column_text(diagnostics_statement.get(), 4)});
            next_diagnostic = std::max(next_diagnostic, sequence + 1);
        }
    }
    return true;
}

bool asset_manager::implementation::persist_record(const record& value)
{
    if (!database || value.virtual_asset) return true;
    sqlite_statement statement(
        database, "INSERT INTO assets(guid,type,importer,source_path,source_hash,dependency_hash,state,"
                  "residency,generation,revision,importer_version,imported_version,source_missing,"
                  "has_last_good,modified,file_size) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                  "ON CONFLICT(guid) DO UPDATE SET type=excluded.type,importer=excluded.importer,"
                  "source_path=excluded.source_path,source_hash=excluded.source_hash,"
                  "dependency_hash=excluded.dependency_hash,state=excluded.state,residency=excluded.residency,"
                  "generation=excluded.generation,revision=excluded.revision,"
                  "importer_version=excluded.importer_version,imported_version=excluded.imported_version,"
                  "source_missing=excluded.source_missing,has_last_good=excluded.has_last_good,"
                  "modified=excluded.modified,file_size=excluded.file_size;");
    if (!statement) return false;
    const auto& snapshot = value.snapshot;
    bind_text(statement.get(), 1, to_string(snapshot.guid));
    bind_text(statement.get(), 2, to_string(snapshot.type));
    bind_text(statement.get(), 3, to_string(snapshot.importer));
    bind_text(statement.get(), 4, normalize_asset_path(snapshot.source_path));
    bind_text(statement.get(), 5, to_string(snapshot.source_hash));
    bind_text(statement.get(), 6, to_string(snapshot.dependency_hash));
    sqlite3_bind_int(statement.get(), 7, static_cast<int>(snapshot.state));
    sqlite3_bind_int(statement.get(), 8, static_cast<int>(snapshot.residency));
    sqlite3_bind_int64(statement.get(), 9, static_cast<sqlite3_int64>(snapshot.generation));
    sqlite3_bind_int64(statement.get(), 10, static_cast<sqlite3_int64>(snapshot.revision));
    sqlite3_bind_int(statement.get(), 11, static_cast<int>(snapshot.importer_version));
    sqlite3_bind_int(statement.get(), 12, static_cast<int>(snapshot.imported_version));
    sqlite3_bind_int(statement.get(), 13, snapshot.source_missing ? 1 : 0);
    sqlite3_bind_int(statement.get(), 14, snapshot.has_last_good ? 1 : 0);
    sqlite3_bind_int64(statement.get(), 15, file_time_value(value.modified));
    sqlite3_bind_int64(statement.get(), 16, static_cast<sqlite3_int64>(value.file_size));
    if (sqlite3_step(statement.get()) != SQLITE_DONE) return false;

    execute(database, "BEGIN;");
    sqlite_statement remove_subassets(database, "DELETE FROM subassets WHERE asset_guid=?;");
    bind_text(remove_subassets.get(), 1, to_string(snapshot.guid));
    sqlite3_step(remove_subassets.get());
    sqlite_statement insert_subasset(database,
                                     "INSERT INTO subassets(asset_guid,persistent_key,guid,type,name,tombstoned)"
                                     " VALUES(?,?,?,?,?,?);");
    for (const auto& subasset : snapshot.subassets)
    {
        sqlite3_reset(insert_subasset.get());
        sqlite3_clear_bindings(insert_subasset.get());
        bind_text(insert_subasset.get(), 1, to_string(snapshot.guid));
        bind_text(insert_subasset.get(), 2, subasset.persistent_key);
        bind_text(insert_subasset.get(), 3, to_string(subasset.guid));
        bind_text(insert_subasset.get(), 4, to_string(subasset.type));
        bind_text(insert_subasset.get(), 5, subasset.name);
        sqlite3_bind_int(insert_subasset.get(), 6, subasset.tombstoned ? 1 : 0);
        sqlite3_step(insert_subasset.get());
    }
    sqlite_statement remove_diagnostics(database, "DELETE FROM diagnostics WHERE asset_guid=?;");
    bind_text(remove_diagnostics.get(), 1, to_string(snapshot.guid));
    sqlite3_step(remove_diagnostics.get());
    sqlite_statement insert_diagnostic(
        database, "INSERT INTO diagnostics(asset_guid,sequence,severity,category,message) VALUES(?,?,?,?,?);");
    for (const auto& diagnostic : snapshot.diagnostics)
    {
        sqlite3_reset(insert_diagnostic.get());
        sqlite3_clear_bindings(insert_diagnostic.get());
        bind_text(insert_diagnostic.get(), 1, to_string(snapshot.guid));
        sqlite3_bind_int64(insert_diagnostic.get(), 2, static_cast<sqlite3_int64>(diagnostic.sequence));
        sqlite3_bind_int(insert_diagnostic.get(), 3, static_cast<int>(diagnostic.severity));
        bind_text(insert_diagnostic.get(), 4, diagnostic.category);
        bind_text(insert_diagnostic.get(), 5, diagnostic.message);
        sqlite3_step(insert_diagnostic.get());
    }
    execute(database, "COMMIT;");
    return true;
}

void asset_manager::implementation::persist_dependencies(const record& value)
{
    if (!database || value.virtual_asset) return;
    execute(database, "BEGIN;");
    sqlite_statement remove(database, "DELETE FROM dependencies WHERE asset_guid=?;");
    bind_text(remove.get(), 1, to_string(value.snapshot.guid));
    sqlite3_step(remove.get());
    sqlite_statement insert(database, "INSERT OR IGNORE INTO dependencies(asset_guid,dependency_guid) VALUES(?,?);");
    for (asset_guid dependency : value.snapshot.dependencies)
    {
        sqlite3_reset(insert.get());
        sqlite3_clear_bindings(insert.get());
        bind_text(insert.get(), 1, to_string(value.snapshot.guid));
        bind_text(insert.get(), 2, to_string(dependency));
        sqlite3_step(insert.get());
    }
    execute(database, "COMMIT;");
}

void asset_manager::implementation::persist_artifacts(const record& value)
{
    if (!database || value.virtual_asset) return;
    execute(database, "BEGIN;");
    sqlite_statement remove(database, "DELETE FROM artifacts WHERE asset_guid=?;");
    bind_text(remove.get(), 1, to_string(value.snapshot.guid));
    sqlite3_step(remove.get());
    sqlite_statement insert(
        database, "INSERT INTO artifacts(asset_guid,name,path,content_hash,size,residency) VALUES(?,?,?,?,?,?);");
    for (const auto& artifact : value.snapshot.artifacts)
    {
        sqlite3_reset(insert.get());
        sqlite3_clear_bindings(insert.get());
        bind_text(insert.get(), 1, to_string(value.snapshot.guid));
        bind_text(insert.get(), 2, artifact.name);
        bind_text(insert.get(), 3, normalize_asset_path(artifact.path));
        bind_text(insert.get(), 4, to_string(artifact.content_hash));
        sqlite3_bind_int64(insert.get(), 5, static_cast<sqlite3_int64>(artifact.size));
        sqlite3_bind_int(insert.get(), 6, static_cast<int>(artifact.residency));
        sqlite3_step(insert.get());
    }
    execute(database, "COMMIT;");
}

void asset_manager::implementation::clear_tombstone(asset_guid guid)
{
    if (!database) return;
    sqlite_statement remove(database, "DELETE FROM tombstones WHERE guid=?;");
    bind_text(remove.get(), 1, to_string(guid));
    sqlite3_step(remove.get());
}

void asset_manager::implementation::persist_tombstone(const record& value)
{
    if (!database) return;
    sqlite_statement tombstone(
        database, "INSERT OR REPLACE INTO tombstones(guid,type,last_path,deleted_revision) VALUES(?,?,?,?);");
    bind_text(tombstone.get(), 1, to_string(value.snapshot.guid));
    bind_text(tombstone.get(), 2, to_string(value.snapshot.type));
    bind_text(tombstone.get(), 3, normalize_asset_path(value.snapshot.source_path));
    sqlite3_bind_int64(tombstone.get(), 4, static_cast<sqlite3_int64>(value.snapshot.revision));
    sqlite3_step(tombstone.get());
}

} // namespace arc::assets
