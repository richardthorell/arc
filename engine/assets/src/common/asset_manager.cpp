#include <arc/assets/assets.h>

#include <nlohmann/json.hpp>
#include <sqlite3.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace arc::assets
{
namespace
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

class sqlite_statement
{
public:
    sqlite_statement(sqlite3* database, const char* sql)
    {
        sqlite3_prepare_v2(database, sql, -1, &statement_, nullptr);
    }
    ~sqlite_statement()
    {
        if (statement_) sqlite3_finalize(statement_);
    }
    sqlite_statement(const sqlite_statement&) = delete;
    sqlite_statement& operator=(const sqlite_statement&) = delete;
    sqlite3_stmt* get() const noexcept
    {
        return statement_;
    }
    explicit operator bool() const noexcept
    {
        return statement_ != nullptr;
    }

private:
    sqlite3_stmt* statement_{};
};

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
    sqlite3_bind_text(statement, index, value.data(), static_cast<int>(value.size()), SQLITE_TRANSIENT);
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
        const auto resolved = context.metadata.type == asset_types::material
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

    std::filesystem::path resolved;
    if (normalized_text == "assets" || normalized_text.starts_with("assets/"))
        resolved = context.project_root / authored;
    else if (context.metadata.type == asset_types::material)
        resolved = context.project_root / "assets" / authored;
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
                                               asset_types::shader));
        cursor = end + 1;
    }
}

class source_blob_importer final : public asset_importer
{
public:
    source_blob_importer(asset_importer_id id, asset_type_id type, std::string name,
                         std::vector<std::string> extensions)
    {
        descriptor_.id = id;
        descriptor_.name = std::move(name);
        descriptor_.extensions = std::move(extensions);
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
            context.metadata.type == asset_types::material || context.source_path.extension() == ".gltf")
        {
            const auto document = nlohmann::json::parse(reinterpret_cast<const char*>(context.source_bytes.data()),
                                                        reinterpret_cast<const char*>(context.source_bytes.data()) +
                                                            context.source_bytes.size(),
                                                        nullptr, false);
            if (!document.is_discarded()) collect_json_dependencies(document, context, result.dependencies);
        }
        else if (context.metadata.type == asset_types::shader)
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
    result.push_back(std::make_unique<source_blob_importer>(
        importer_ids::shader, asset_types::shader, "ARC Shader",
        std::vector<std::string>{".slang", ".glsl", ".vert", ".frag", ".comp", ".hlsl", ".inc"}));
    result.push_back(std::make_unique<source_blob_importer>(
        importer_ids::texture, asset_types::texture_2d, "Texture",
        std::vector<std::string>{".png", ".jpg", ".jpeg", ".dds", ".tga", ".bmp", ".ktx", ".ktx2"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::environment, asset_types::environment,
                                                            "Environment", std::vector<std::string>{".hdr", ".exr"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::gltf, asset_types::imported_scene, "glTF",
                                                            std::vector<std::string>{".glb", ".gltf"}));
    result.push_back(std::make_unique<source_blob_importer>(importer_ids::fbx, asset_types::imported_scene, "FBX",
                                                            std::vector<std::string>{".fbx"}));
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

} // namespace

struct asset_manager::implementation
{
    struct record
    {
        asset_snapshot snapshot;
        asset_source_metadata metadata;
        std::filesystem::path absolute_path;
        std::filesystem::file_time_type modified{};
        std::uint64_t file_size{};
        std::shared_ptr<detail::asset_slot> slot = std::make_shared<detail::asset_slot>();
        jobs::job_handle active_import;
        jobs::cancellation_source import_cancellation;
        std::chrono::steady_clock::time_point last_used{};
        std::filesystem::file_time_type pending_modified{};
        std::uint64_t pending_file_size{};
        std::chrono::steady_clock::time_point pending_since{};
        asset_residency requested_residency{asset_residency::cpu};
        bool pending_source_change{};
        bool virtual_asset{};
    };

    asset_manager_config config;
    jobs::job_system* jobs{};
    io::async_file_service* files{};
    memory::memory_system* memory{};
    std::unique_ptr<memory::streaming_heap> streaming;
    std::uint64_t pressure_handler{};
    mutable std::shared_mutex mutex;
    sqlite3* database{};
    std::unordered_map<asset_guid, record, asset_guid_hash> records;
    std::unordered_map<std::string, asset_guid> paths;
    std::unordered_map<asset_importer_id, std::unique_ptr<asset_importer>, asset_importer_id_hash> importers;
    std::unordered_map<asset_type_id, asset_guid, asset_type_id_hash> fallbacks;
    std::vector<missing_asset_reference> missing_references;
    std::vector<asset_event> events;
    std::unordered_map<std::uint64_t, asset_event_callback> subscribers;
    std::uint64_t next_subscription{1};
    std::uint64_t next_event{1};
    std::uint64_t revision{};
    std::uint64_t next_diagnostic{1};
    std::chrono::steady_clock::time_point next_poll{};
    bool started{};

    implementation(asset_manager_config value, jobs::job_system& scheduler, io::async_file_service& file_service,
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

    ~implementation()
    {
        if (pressure_handler) memory->remove_pressure_handler(pressure_handler);
        if (database) sqlite3_close(database);
    }

    bool managed_source_path(const std::filesystem::path& path) const
    {
        return path_within(config.asset_root, path) ||
               std::any_of(config.additional_source_roots.begin(), config.additional_source_roots.end(),
                           [&](const auto& root) { return path_within(root, path); });
    }

    void emit(asset_event_kind kind, asset_guid guid, asset_state state, std::string message, float progress = 0.0f)
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

    asset_diagnostic diagnostic(asset_guid guid, asset_diagnostic_severity severity, std::string category,
                                std::string message)
    {
        return {.sequence = next_diagnostic++,
                .severity = severity,
                .guid = guid,
                .category = std::move(category),
                .message = std::move(message)};
    }

    bool open_database(std::string& error)
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

    bool rebuild_database(std::string& error)
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

    bool load_database(std::string& error)
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
        sqlite_statement subassets_statement(
            database, "SELECT asset_guid,persistent_key,guid,type,name,tombstoned FROM subassets "
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
                found->second.snapshot.diagnostics.push_back({.sequence = sequence,
                                                              .severity = static_cast<asset_diagnostic_severity>(
                                                                  sqlite3_column_int(diagnostics_statement.get(), 2)),
                                                              .guid = *owner,
                                                              .category = column_text(diagnostics_statement.get(), 3),
                                                              .message = column_text(diagnostics_statement.get(), 4)});
                next_diagnostic = std::max(next_diagnostic, sequence + 1);
            }
        }
        return true;
    }

    bool persist_record(const record& value)
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

    void persist_dependencies(const record& value)
    {
        if (!database || value.virtual_asset) return;
        execute(database, "BEGIN;");
        sqlite_statement remove(database, "DELETE FROM dependencies WHERE asset_guid=?;");
        bind_text(remove.get(), 1, to_string(value.snapshot.guid));
        sqlite3_step(remove.get());
        sqlite_statement insert(database,
                                "INSERT OR IGNORE INTO dependencies(asset_guid,dependency_guid) VALUES(?,?);");
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

    void persist_artifacts(const record& value)
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

    void clear_tombstone(asset_guid guid)
    {
        if (!database) return;
        sqlite_statement remove(database, "DELETE FROM tombstones WHERE guid=?;");
        bind_text(remove.get(), 1, to_string(guid));
        sqlite3_step(remove.get());
    }

    bool dependency_reaches(asset_guid current, asset_guid target,
                            std::unordered_set<asset_guid, asset_guid_hash>& visited) const
    {
        if (current == target) return true;
        if (!visited.insert(current).second) return false;
        const auto found = records.find(current);
        if (found == records.end()) return false;
        return std::any_of(found->second.snapshot.dependencies.begin(), found->second.snapshot.dependencies.end(),
                           [&](asset_guid dependency) { return dependency_reaches(dependency, target, visited); });
    }

    void mark_reverse_stale(asset_guid guid, std::string_view reason)
    {
        const auto found = records.find(guid);
        if (found == records.end()) return;
        for (asset_guid dependent : found->second.snapshot.reverse_dependencies)
        {
            const auto dependent_found = records.find(dependent);
            if (dependent_found == records.end() || dependent_found->second.snapshot.state == asset_state::stale)
                continue;
            dependent_found->second.snapshot.state = asset_state::stale;
            dependent_found->second.snapshot.revision = ++revision;
            dependent_found->second.snapshot.diagnostics.push_back(
                diagnostic(dependent, asset_diagnostic_severity::information, "dependency",
                           "Dependency changed: " + std::string(reason)));
            persist_record(dependent_found->second);
            emit(asset_event_kind::dependencies_changed, dependent, asset_state::stale,
                 "Asset dependency became stale");
            mark_reverse_stale(dependent, reason);
        }
    }

    jobs::job_handle ensure_import(asset_guid guid, asset_streaming_priority priority,
                                   jobs::cancellation_token cancellation,
                                   asset_residency requested_residency = asset_residency::cpu)
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
            value.snapshot.diagnostics.push_back(diagnostic(guid, asset_diagnostic_severity::error, "import",
                                                            "No importer is registered for this asset"));
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
                                          .message =
                                              "An imported dependency is missing, cyclic, mismatched, or failed"};
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
                        const auto old = std::find_if(target.metadata.subassets.begin(),
                                                      target.metadata.subassets.end(), [&](const auto& value)
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
                            target.snapshot.diagnostics.push_back(diagnostic(guid, asset_diagnostic_severity::warning,
                                                                             "metadata", saved.error().message));
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
                    hash_bytes(std::as_bytes(
                        std::span(metadata.canonical_settings.data(), metadata.canonical_settings.size()))),
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
};

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
            const auto original_error = error;
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
        if (implementation_->database)
        {
            sqlite_statement tombstone(
                implementation_->database,
                "INSERT OR REPLACE INTO tombstones(guid,type,last_path,deleted_revision) VALUES(?,?,?,?);");
            bind_text(tombstone.get(), 1, to_string(guid));
            bind_text(tombstone.get(), 2, to_string(value.snapshot.type));
            bind_text(tombstone.get(), 3, normalize_asset_path(value.snapshot.source_path));
            sqlite3_bind_int64(tombstone.get(), 4, static_cast<sqlite3_int64>(value.snapshot.revision));
            sqlite3_step(tombstone.get());
        }
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
        if (!needle.empty() && path_key(value.snapshot.source_path).find(needle) == std::string::npos) continue;
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
