#include <arc/assets/asset_search.h>

#include <sqlite3.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <sstream>
#include <string_view>

namespace arc::assets
{
namespace
{

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

    [[nodiscard]] sqlite3_stmt* get() const noexcept
    {
        return statement_;
    }

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return statement_ != nullptr;
    }

private:
    sqlite3_stmt* statement_{};
};

void bind_text(sqlite3_stmt* statement, int index, std::string_view value)
{
    sqlite3_bind_text(statement, index, value.data(), static_cast<int>(value.size()),
                      SQLITE_TRANSIENT); // NOLINT(performance-no-int-to-ptr)
}

std::string column_text(sqlite3_stmt* statement, int column)
{
    const auto* value = sqlite3_column_text(statement, column);
    return value ? reinterpret_cast<const char*>(value) : std::string{};
}

asset_status sqlite_failure(sqlite3* database, const std::filesystem::path& path, std::string message)
{
    if (database)
    {
        message += ": ";
        message += sqlite3_errmsg(database);
    }
    return asset_status::failure({.code = asset_error_code::io_failed, .path = path, .message = std::move(message)});
}

bool execute(sqlite3* database, const char* sql)
{
    char* error{};
    const auto result = sqlite3_exec(database, sql, nullptr, nullptr, &error);
    if (error) sqlite3_free(error);
    return result == SQLITE_OK;
}

std::string lowercase(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return result;
}

bool contains_provider(std::span<const std::string> providers, std::string_view provider)
{
    return providers.empty() || std::find(providers.begin(), providers.end(), provider) != providers.end();
}

std::string join_values(std::span<const std::string> values)
{
    std::string result;
    for (const auto& value : values)
    {
        if (!result.empty()) result.push_back(' ');
        result += value;
    }
    return result;
}

std::string join_metadata(std::span<const asset_search_metadata_value> metadata)
{
    std::string result;
    for (const auto& entry : metadata)
    {
        if (!result.empty()) result.push_back(' ');
        result += entry.key;
        result.push_back(' ');
        result += entry.value;
    }
    return result;
}

std::string fts_query(std::string_view text)
{
    std::vector<std::string> tokens;
    std::string token;
    for (const unsigned char character : text)
    {
        if (std::isalnum(character) || character == '_' || character == '-')
        {
            token.push_back(static_cast<char>(character));
            continue;
        }
        if (!token.empty())
        {
            tokens.push_back(std::move(token));
            token.clear();
        }
    }
    if (!token.empty()) tokens.push_back(std::move(token));
    if (tokens.empty()) return {};

    std::string result;
    for (const auto& value : tokens)
    {
        if (!result.empty()) result += " AND ";
        result.push_back('"');
        for (const auto character : value)
        {
            if (character == '"') result += """";
            else result.push_back(character);
        }
        result += ""*";
    }
    return result;
}

struct baseline_match
{
    float score{};
    asset_search_match_kind kind{asset_search_match_kind::lexical};
    std::string detail;
    bool matched{};
};

baseline_match match_text(std::string_view query, std::string_view display_name, std::string_view title,
                          std::string_view logical_path, std::string_view description,
                          std::string_view authored_tags, std::string_view generated_tags,
                          std::string_view categories, std::string_view metadata)
{
    if (query.empty()) return {.matched = true};

    const auto needle = lowercase(query);
    const auto name = lowercase(display_name);
    const auto normalized_title = lowercase(title);
    const auto path = lowercase(logical_path);
    const auto desc = lowercase(description);
    const auto authored = lowercase(authored_tags);
    const auto generated = lowercase(generated_tags);
    const auto normalized_categories = lowercase(categories);
    const auto normalized_metadata = lowercase(metadata);

    if (name == needle || normalized_title == needle)
        return {.score = 1.0f, .kind = asset_search_match_kind::exact_name, .detail = "name", .matched = true};
    if (name.starts_with(needle) || normalized_title.starts_with(needle))
        return {.score = 0.9f, .kind = asset_search_match_kind::prefix, .detail = "name", .matched = true};
    if (name.find(needle) != std::string::npos || normalized_title.find(needle) != std::string::npos)
        return {.score = 0.8f, .kind = asset_search_match_kind::lexical, .detail = "name", .matched = true};
    if (authored.find(needle) != std::string::npos)
        return {.score = 0.75f, .kind = asset_search_match_kind::metadata, .detail = "authored tag", .matched = true};
    if (generated.find(needle) != std::string::npos)
        return {.score = 0.7f, .kind = asset_search_match_kind::metadata, .detail = "generated tag", .matched = true};
    if (normalized_categories.find(needle) != std::string::npos)
        return {.score = 0.68f, .kind = asset_search_match_kind::metadata, .detail = "category", .matched = true};
    if (path.find(needle) != std::string::npos)
        return {.score = 0.6f, .kind = asset_search_match_kind::lexical, .detail = "path", .matched = true};
    if (desc.find(needle) != std::string::npos)
        return {.score = 0.55f, .kind = asset_search_match_kind::lexical, .detail = "description", .matched = true};
    if (normalized_metadata.find(needle) != std::string::npos)
        return {.score = 0.5f, .kind = asset_search_match_kind::metadata, .detail = "metadata", .matched = true};
    // FTS can match tokenized/prefix combinations that do not appear as one
    // contiguous phrase in any single field.
    return {.score = 0.45f, .kind = asset_search_match_kind::lexical, .detail = "tokens", .matched = true};
}

} // namespace

struct persistent_asset_search_index::implementation
{
    std::filesystem::path path;
    sqlite3* database{};

    ~implementation()
    {
        if (database) sqlite3_close(database);
    }
};

namespace
{

asset_status create_schema(sqlite3* database, const std::filesystem::path& path)
{
    const char* schema =
        "BEGIN;"
        "CREATE TABLE IF NOT EXISTS search_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);"
        "CREATE TABLE IF NOT EXISTS search_documents("
        "guid TEXT PRIMARY KEY,type TEXT NOT NULL,display_name TEXT NOT NULL,logical_path TEXT NOT NULL,"
        "active_provider TEXT NOT NULL,title TEXT NOT NULL,description TEXT NOT NULL,"
        "source_generation INTEGER NOT NULL,source_revision INTEGER NOT NULL,"
        "document_schema_version INTEGER NOT NULL);"
        "CREATE INDEX IF NOT EXISTS search_documents_type ON search_documents(type);"
        "CREATE INDEX IF NOT EXISTS search_documents_provider ON search_documents(active_provider);"
        "CREATE INDEX IF NOT EXISTS search_documents_name ON search_documents(display_name COLLATE NOCASE);"
        "CREATE TABLE IF NOT EXISTS search_authored_tags("
        "guid TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(guid,value));"
        "CREATE TABLE IF NOT EXISTS search_generated_tags("
        "guid TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(guid,value));"
        "CREATE TABLE IF NOT EXISTS search_categories("
        "guid TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(guid,value));"
        "CREATE TABLE IF NOT EXISTS search_metadata("
        "guid TEXT NOT NULL,key TEXT NOT NULL,value TEXT NOT NULL,origin INTEGER NOT NULL,"
        "PRIMARY KEY(guid,key,value,origin));"
        "CREATE VIRTUAL TABLE IF NOT EXISTS search_fts USING fts5("
        "guid UNINDEXED,display_name,title,logical_path,description,"
        "authored_tags,generated_tags,categories,metadata,"
        "tokenize='unicode61 remove_diacritics 2');"
        "INSERT OR IGNORE INTO search_meta(key,value) VALUES('schema_version','2');"
        "COMMIT;";
    if (!execute(database, schema)) return sqlite_failure(database, path, "Could not create search index schema");
    return asset_status::success();
}

asset_status clear_schema(sqlite3* database, const std::filesystem::path& path)
{
    const char* sql = "BEGIN;"
                      "DROP TABLE IF EXISTS search_fts;"
                      "DROP TABLE IF EXISTS search_metadata;"
                      "DROP TABLE IF EXISTS search_categories;"
                      "DROP TABLE IF EXISTS search_generated_tags;"
                      "DROP TABLE IF EXISTS search_authored_tags;"
                      "DROP TABLE IF EXISTS search_documents;"
                      "DROP TABLE IF EXISTS search_meta;"
                      "COMMIT;";
    if (!execute(database, sql)) return sqlite_failure(database, path, "Could not reset search index schema");
    return create_schema(database, path);
}

asset_status replace_values(sqlite3* database, const std::filesystem::path& path, std::string_view table,
                            asset_guid guid, std::span<const std::string> values)
{
    const auto guid_text = to_string(guid);
    const std::string delete_sql = "DELETE FROM " + std::string(table) + " WHERE guid=?;";
    sqlite_statement erase(database, delete_sql.c_str());
    if (!erase) return sqlite_failure(database, path, "Could not prepare search metadata delete");
    bind_text(erase.get(), 1, guid_text);
    if (sqlite3_step(erase.get()) != SQLITE_DONE)
        return sqlite_failure(database, path, "Could not delete old search metadata");

    const std::string insert_sql = "INSERT INTO " + std::string(table) + "(guid,value) VALUES(?,?);";
    sqlite_statement insert(database, insert_sql.c_str());
    if (!insert) return sqlite_failure(database, path, "Could not prepare search metadata insert");
    for (const auto& value : values)
    {
        sqlite3_reset(insert.get());
        sqlite3_clear_bindings(insert.get());
        bind_text(insert.get(), 1, guid_text);
        bind_text(insert.get(), 2, value);
        if (sqlite3_step(insert.get()) != SQLITE_DONE)
            return sqlite_failure(database, path, "Could not write search metadata");
    }
    return asset_status::success();
}

asset_status write_document(sqlite3* database, const std::filesystem::path& path, const asset_search_document& document)
{
    sqlite_statement statement(
        database, "INSERT INTO search_documents("
                  "guid,type,display_name,logical_path,active_provider,title,description,"
                  "source_generation,source_revision,document_schema_version)"
                  "VALUES(?,?,?,?,?,?,?,?,?,?)"
                  "ON CONFLICT(guid) DO UPDATE SET "
                  "type=excluded.type,display_name=excluded.display_name,logical_path=excluded.logical_path,"
                  "active_provider=excluded.active_provider,title=excluded.title,description=excluded.description,"
                  "source_generation=excluded.source_generation,source_revision=excluded.source_revision,"
                  "document_schema_version=excluded.document_schema_version;");
    if (!statement) return sqlite_failure(database, path, "Could not prepare search document upsert");

    bind_text(statement.get(), 1, to_string(document.guid));
    bind_text(statement.get(), 2, to_string(document.type));
    bind_text(statement.get(), 3, document.display_name);
    bind_text(statement.get(), 4, document.logical_path);
    bind_text(statement.get(), 5, document.active_provider);
    bind_text(statement.get(), 6, document.title);
    bind_text(statement.get(), 7, document.description);
    sqlite3_bind_int64(statement.get(), 8, static_cast<sqlite3_int64>(document.source_generation));
    sqlite3_bind_int64(statement.get(), 9, static_cast<sqlite3_int64>(document.source_revision));
    sqlite3_bind_int(statement.get(), 10, static_cast<int>(document.schema_version));
    if (sqlite3_step(statement.get()) != SQLITE_DONE)
        return sqlite_failure(database, path, "Could not write search document");

    auto status = replace_values(database, path, "search_authored_tags", document.guid, document.authored_tags);
    if (!status) return status;
    status = replace_values(database, path, "search_generated_tags", document.guid, document.generated_tags);
    if (!status) return status;
    status = replace_values(database, path, "search_categories", document.guid, document.categories);
    if (!status) return status;

    sqlite_statement erase_metadata(database, "DELETE FROM search_metadata WHERE guid=?;");
    if (!erase_metadata) return sqlite_failure(database, path, "Could not prepare metadata reset");
    bind_text(erase_metadata.get(), 1, to_string(document.guid));
    if (sqlite3_step(erase_metadata.get()) != SQLITE_DONE)
        return sqlite_failure(database, path, "Could not clear search metadata");

    sqlite_statement insert_metadata(database, "INSERT INTO search_metadata(guid,key,value,origin) VALUES(?,?,?,?);");
    if (!insert_metadata) return sqlite_failure(database, path, "Could not prepare metadata insert");
    for (const auto& metadata : document.metadata)
    {
        sqlite3_reset(insert_metadata.get());
        sqlite3_clear_bindings(insert_metadata.get());
        bind_text(insert_metadata.get(), 1, to_string(document.guid));
        bind_text(insert_metadata.get(), 2, metadata.key);
        bind_text(insert_metadata.get(), 3, metadata.value);
        sqlite3_bind_int(insert_metadata.get(), 4, static_cast<int>(metadata.origin));
        if (sqlite3_step(insert_metadata.get()) != SQLITE_DONE)
            return sqlite_failure(database, path, "Could not write search metadata");
    }

    sqlite_statement erase_fts(database, "DELETE FROM search_fts WHERE guid=?;");
    if (!erase_fts) return sqlite_failure(database, path, "Could not prepare FTS reset");
    bind_text(erase_fts.get(), 1, to_string(document.guid));
    if (sqlite3_step(erase_fts.get()) != SQLITE_DONE)
        return sqlite_failure(database, path, "Could not clear FTS search document");

    sqlite_statement insert_fts(
        database,
        "INSERT INTO search_fts(guid,display_name,title,logical_path,description,"
        "authored_tags,generated_tags,categories,metadata) VALUES(?,?,?,?,?,?,?,?,?);");
    if (!insert_fts) return sqlite_failure(database, path, "Could not prepare FTS search document");
    bind_text(insert_fts.get(), 1, to_string(document.guid));
    bind_text(insert_fts.get(), 2, document.display_name);
    bind_text(insert_fts.get(), 3, document.title);
    bind_text(insert_fts.get(), 4, document.logical_path);
    bind_text(insert_fts.get(), 5, document.description);
    const auto authored = join_values(document.authored_tags);
    const auto generated = join_values(document.generated_tags);
    const auto category_text = join_values(document.categories);
    const auto metadata_text = join_metadata(document.metadata);
    bind_text(insert_fts.get(), 6, authored);
    bind_text(insert_fts.get(), 7, generated);
    bind_text(insert_fts.get(), 8, category_text);
    bind_text(insert_fts.get(), 9, metadata_text);
    if (sqlite3_step(insert_fts.get()) != SQLITE_DONE)
        return sqlite_failure(database, path, "Could not write FTS search document");

    return asset_status::success();
}

} // namespace

persistent_asset_search_index::persistent_asset_search_index(std::unique_ptr<implementation> implementation)
    : implementation_(std::move(implementation))
{
}

persistent_asset_search_index::~persistent_asset_search_index() = default;

persistent_asset_search_index::open_result persistent_asset_search_index::open(std::filesystem::path database_path)
{
    std::error_code filesystem_error;
    if (!database_path.parent_path().empty())
        std::filesystem::create_directories(database_path.parent_path(), filesystem_error);
    if (filesystem_error)
        return open_result::failure({.code = asset_error_code::io_failed,
                                     .path = database_path,
                                     .message = "Could not create search index directory"});

    auto state = std::make_unique<implementation>();
    state->path = std::move(database_path);

    const auto open_database = [&]() -> bool
    {
        return sqlite3_open_v2(state->path.string().c_str(), &state->database,
                               SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX,
                               nullptr) == SQLITE_OK;
    };

    if (!open_database())
        return open_result::failure(
            {.code = asset_error_code::io_failed, .path = state->path, .message = "Could not open search index"});

    execute(state->database, "PRAGMA journal_mode=WAL;");
    execute(state->database, "PRAGMA synchronous=NORMAL;");

    bool healthy = true;
    {
        sqlite_statement integrity(state->database, "PRAGMA quick_check;");
        healthy = integrity && sqlite3_step(integrity.get()) == SQLITE_ROW && column_text(integrity.get(), 0) == "ok";
    }
    if (!healthy)
    {
        sqlite3_close(state->database);
        state->database = nullptr;
        const auto corrupt = state->path.string() + ".corrupt-" +
                             std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::filesystem::rename(state->path, corrupt, filesystem_error);
        if (filesystem_error)
        {
            filesystem_error.clear();
            std::filesystem::remove(state->path, filesystem_error);
        }
        if (!open_database())
            return open_result::failure({.code = asset_error_code::io_failed,
                                         .path = state->path,
                                         .message = "Could not recreate corrupt search index"});
    }

    auto status = create_schema(state->database, state->path);
    if (!status) return open_result::failure(std::move(status).error());

    std::uint32_t version{};
    {
        sqlite_statement statement(state->database, "SELECT value FROM search_meta WHERE key='schema_version';");
        if (!statement || sqlite3_step(statement.get()) != SQLITE_ROW)
            return open_result::failure({.code = asset_error_code::invalid_metadata,
                                         .path = state->path,
                                         .message = "Search index has no schema version"});
        const auto text = column_text(statement.get(), 0);
        const auto parsed = std::from_chars(text.data(), text.data() + text.size(), version);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) version = 0;
    }

    if (version != asset_search_index_schema_version)
    {
        status = clear_schema(state->database, state->path);
        if (!status) return open_result::failure(std::move(status).error());
    }

    return open_result::success(
        std::unique_ptr<persistent_asset_search_index>(new persistent_asset_search_index(std::move(state))));
}

asset_status persistent_asset_search_index::reset(std::span<const asset_search_document> documents)
{
    if (!implementation_ || !implementation_->database)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Search index is not open"});

    if (!execute(implementation_->database, "BEGIN IMMEDIATE;"
                                            "DELETE FROM search_fts;"
                                            "DELETE FROM search_metadata;"
                                            "DELETE FROM search_categories;"
                                            "DELETE FROM search_generated_tags;"
                                            "DELETE FROM search_authored_tags;"
                                            "DELETE FROM search_documents;"
                                            "DELETE FROM search_meta WHERE key='source_change_cursor';"))
        return sqlite_failure(implementation_->database, implementation_->path, "Could not reset search index");

    for (const auto& document : documents)
    {
        if (!document.guid.valid() || !document.type.valid() ||
            document.schema_version != asset_search_document_schema_version)
        {
            execute(implementation_->database, "ROLLBACK;");
            return asset_status::failure({.code = asset_error_code::invalid_metadata,
                                          .guid = document.guid,
                                          .message = "Search rebuild contains an incompatible document"});
        }
        auto status = write_document(implementation_->database, implementation_->path, document);
        if (!status)
        {
            execute(implementation_->database, "ROLLBACK;");
            return status;
        }
    }

    if (!execute(implementation_->database, "COMMIT;"))
    {
        execute(implementation_->database, "ROLLBACK;");
        return sqlite_failure(implementation_->database, implementation_->path, "Could not commit search rebuild");
    }
    return asset_status::success();
}

asset_status persistent_asset_search_index::upsert(asset_search_document document)
{
    if (!implementation_ || !implementation_->database)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Search index is not open"});
    if (!document.guid.valid() || !document.type.valid())
        return asset_status::failure({.code = asset_error_code::invalid_metadata,
                                      .guid = document.guid,
                                      .message = "Search document identity is incomplete"});
    if (document.schema_version != asset_search_document_schema_version)
        return asset_status::failure({.code = asset_error_code::invalid_metadata,
                                      .guid = document.guid,
                                      .message = "Search document schema is incompatible"});

    if (!execute(implementation_->database, "BEGIN IMMEDIATE;"))
        return sqlite_failure(implementation_->database, implementation_->path, "Could not begin search update");

    {
        sqlite_statement existing(implementation_->database,
                                  "SELECT source_generation,source_revision FROM search_documents WHERE guid=?;");
        if (!existing)
        {
            execute(implementation_->database, "ROLLBACK;");
            return sqlite_failure(implementation_->database, implementation_->path,
                                  "Could not inspect existing search document");
        }
        bind_text(existing.get(), 1, to_string(document.guid));
        if (sqlite3_step(existing.get()) == SQLITE_ROW)
        {
            const auto generation = static_cast<std::uint64_t>(sqlite3_column_int64(existing.get(), 0));
            const auto revision = static_cast<std::uint64_t>(sqlite3_column_int64(existing.get(), 1));
            if (revision > document.source_revision ||
                (revision == document.source_revision && generation > document.source_generation))
            {
                execute(implementation_->database, "ROLLBACK;");
                return asset_status::failure({.code = asset_error_code::invalid_metadata,
                                              .guid = document.guid,
                                              .path = implementation_->path,
                                              .message = "Search document is older than the cached source revision"});
            }
        }
    }

    auto status = write_document(implementation_->database, implementation_->path, document);
    if (!status)
    {
        execute(implementation_->database, "ROLLBACK;");
        return status;
    }
    if (!execute(implementation_->database, "COMMIT;"))
    {
        execute(implementation_->database, "ROLLBACK;");
        return sqlite_failure(implementation_->database, implementation_->path, "Could not commit search update");
    }
    return asset_status::success();
}

asset_status persistent_asset_search_index::erase(asset_guid guid)
{
    if (!implementation_ || !implementation_->database)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Search index is not open"});

    if (!execute(implementation_->database, "BEGIN IMMEDIATE;"))
        return sqlite_failure(implementation_->database, implementation_->path, "Could not begin search delete");

    const auto guid_text = to_string(guid);
    for (const auto* table :
         {"search_fts", "search_metadata", "search_categories", "search_generated_tags", "search_authored_tags",
          "search_documents"})
    {
        const std::string sql = "DELETE FROM " + std::string(table) + " WHERE guid=?;";
        sqlite_statement statement(implementation_->database, sql.c_str());
        if (!statement)
        {
            execute(implementation_->database, "ROLLBACK;");
            return sqlite_failure(implementation_->database, implementation_->path, "Could not prepare search delete");
        }
        bind_text(statement.get(), 1, guid_text);
        if (sqlite3_step(statement.get()) != SQLITE_DONE)
        {
            execute(implementation_->database, "ROLLBACK;");
            return sqlite_failure(implementation_->database, implementation_->path, "Could not delete search data");
        }
    }

    if (!execute(implementation_->database, "COMMIT;"))
    {
        execute(implementation_->database, "ROLLBACK;");
        return sqlite_failure(implementation_->database, implementation_->path, "Could not commit search delete");
    }
    return asset_status::success();
}

std::optional<std::uint64_t> persistent_asset_search_index::source_change_cursor() const
{
    if (!implementation_ || !implementation_->database) return std::nullopt;

    sqlite_statement statement(implementation_->database,
                               "SELECT value FROM search_meta WHERE key='source_change_cursor';");
    if (!statement || sqlite3_step(statement.get()) != SQLITE_ROW) return std::nullopt;

    const auto text = column_text(statement.get(), 0);
    std::uint64_t cursor{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), cursor);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return std::nullopt;
    return cursor;
}

asset_status persistent_asset_search_index::set_source_change_cursor(std::uint64_t cursor)
{
    if (!implementation_ || !implementation_->database)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Search index is not open"});

    sqlite_statement statement(implementation_->database,
                               "INSERT INTO search_meta(key,value) VALUES('source_change_cursor',?) "
                               "ON CONFLICT(key) DO UPDATE SET value=excluded.value;");
    if (!statement)
        return sqlite_failure(implementation_->database, implementation_->path,
                              "Could not prepare search change checkpoint");

    const auto value = std::to_string(cursor);
    bind_text(statement.get(), 1, value);
    if (sqlite3_step(statement.get()) != SQLITE_DONE)
        return sqlite_failure(implementation_->database, implementation_->path,
                              "Could not persist search change checkpoint");
    return asset_status::success();
}

std::vector<asset_search_result> persistent_asset_search_index::query(const asset_search_query& query_value) const
{
    std::vector<asset_search_result> results;
    if (!implementation_ || !implementation_->database || query_value.limit == 0) return results;

    const bool use_fts = !query_value.text.empty();
    const auto match_query = use_fts ? fts_query(query_value.text) : std::string{};
    if (use_fts && match_query.empty()) return results;

    const char* sql = use_fts
                          ? "SELECT d.guid,d.type,d.display_name,d.logical_path,d.active_provider,d.title,"
                            "d.description,f.authored_tags,f.generated_tags,f.categories,f.metadata "
                            "FROM search_fts f JOIN search_documents d ON d.guid=f.guid "
                            "WHERE search_fts MATCH ?;"
                          : "SELECT d.guid,d.type,d.display_name,d.logical_path,d.active_provider,d.title,"
                            "d.description,'','','','' FROM search_documents d;";
    sqlite_statement statement(implementation_->database, sql);
    if (!statement) return results;
    if (use_fts) bind_text(statement.get(), 1, match_query);

    while (sqlite3_step(statement.get()) == SQLITE_ROW)
    {
        const auto guid = parse_asset_guid(column_text(statement.get(), 0));
        const auto type = parse_asset_type_id(column_text(statement.get(), 1));
        if (!guid || !type) continue;
        if (query_value.filters.type && *query_value.filters.type != *type) continue;

        const auto provider = column_text(statement.get(), 4);
        if (!contains_provider(query_value.filters.provider_ids, provider)) continue;

        baseline_match match{.matched = true};
        if (use_fts)
            match = match_text(query_value.text, column_text(statement.get(), 2), column_text(statement.get(), 5),
                               column_text(statement.get(), 3), column_text(statement.get(), 6),
                               column_text(statement.get(), 7), column_text(statement.get(), 8),
                               column_text(statement.get(), 9), column_text(statement.get(), 10));

        asset_search_result result{.guid = *guid, .score = match.score};
        if (use_fts)
            result.reasons.push_back({.kind = match.kind, .score = match.score, .detail = match.detail});
        results.push_back(std::move(result));
    }

    std::sort(results.begin(), results.end(),
              [](const asset_search_result& lhs, const asset_search_result& rhs)
              {
                  if (lhs.score != rhs.score) return lhs.score > rhs.score;
                  return to_string(lhs.guid) < to_string(rhs.guid);
              });

    const auto first = std::min(query_value.offset, results.size());
    const auto last = std::min(first + query_value.limit, results.size());
    std::vector<asset_search_result> page;
    page.reserve(last - first);
    for (auto index = first; index < last; ++index)
        page.push_back(std::move(results[index]));
    return page;
}

const std::filesystem::path& persistent_asset_search_index::path() const noexcept
{
    static const std::filesystem::path empty;
    return implementation_ ? implementation_->path : empty;
}

} // namespace arc::assets
