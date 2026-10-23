#include <arc/assets/asset_search.h>

#include <algorithm>
#include <charconv>
#include <cctype>

namespace arc::assets
{

namespace
{

std::string lowercase_search_token(std::string_view value)
{
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return result;
}

std::optional<asset_type_id> search_type_from_alias(std::string_view value)
{
    const auto normalized = lowercase_search_token(value);
    if (const auto parsed = parse_asset_type_id(value)) return parsed;

    if (normalized == "scene") return asset_types::scene;
    if (normalized == "prefab") return asset_types::prefab;
    if (normalized == "material" || normalized == "mat") return asset_types::material;
    if (normalized == "material-instance" || normalized == "material_instance" || normalized == "matinst")
        return asset_types::material_instance;
    if (normalized == "material-function" || normalized == "material_function" || normalized == "matfn")
        return asset_types::material_function;
    if (normalized == "texture" || normalized == "texture2d" || normalized == "texture_2d")
        return asset_types::texture_2d;
    if (normalized == "mesh" || normalized == "static-mesh" || normalized == "static_mesh")
        return asset_types::static_mesh;
    if (normalized == "shader") return asset_types::shader;
    if (normalized == "sound") return asset_types::sound;
    if (normalized == "audio" || normalized == "audio-clip" || normalized == "audio_clip")
        return asset_types::audio_clip;
    if (normalized == "animation" || normalized == "animation-clip" || normalized == "animation_clip")
        return asset_types::animation_clip;
    if (normalized == "environment") return asset_types::environment;
    if (normalized == "flow" || normalized == "flow-graph" || normalized == "flow_graph")
        return asset_types::flow_graph;
    return std::nullopt;
}

void append_unique(std::vector<std::string>& values, std::string value)
{
    if (std::find(values.begin(), values.end(), value) == values.end()) values.push_back(std::move(value));
}

bool apply_scope(std::string_view value, std::vector<std::string>& providers)
{
    const auto normalized = lowercase_search_token(value);
    if (normalized == "project")
    {
        append_unique(providers, std::string(asset_provider_ids::project_source));
        return true;
    }
    if (normalized == "builtin" || normalized == "built-in" || normalized == "engine")
    {
        append_unique(providers, std::string(asset_provider_ids::builtin_source));
        append_unique(providers, std::string(asset_provider_ids::virtual_builtin));
        return true;
    }
    return false;
}


std::optional<asset_search_property_kind> search_property_kind(std::string_view key)
{
    const auto normalized = lowercase_search_token(key);
    if (normalized == "triangles" || normalized == "vertices" || normalized == "width" ||
        normalized == "height" || normalized == "roughness" || normalized == "metallic" ||
        normalized == "duration" || normalized == "sample-rate" || normalized == "sample_rate" ||
        normalized == "channels" || normalized == "dependencies" ||
        normalized == "reverse-dependencies" || normalized == "reverse_dependencies" ||
        normalized == "generation" || normalized == "revision")
        return asset_search_property_kind::number;

    if (normalized == "format" || normalized == "encoding" || normalized == "compression" ||
        normalized == "blend-mode" || normalized == "blend_mode")
        return asset_search_property_kind::string;
    return std::nullopt;
}

std::string canonical_property_key(std::string_view key)
{
    auto normalized = lowercase_search_token(key);
    std::replace(normalized.begin(), normalized.end(), '_', '-');
    return normalized;
}

core::result<asset_search_property_filter, std::string>
parse_property_filter(std::string_view key, std::string_view value, asset_search_property_kind kind)
{
    asset_search_property_filter filter;
    filter.key = canonical_property_key(key);
    filter.kind = kind;

    if (kind == asset_search_property_kind::string)
    {
        if (value.starts_with("<") || value.starts_with(">"))
            return core::result<asset_search_property_filter, std::string>::failure(
                "String properties only support equality");
        if (value.starts_with("=")) value.remove_prefix(1);
        if (value.empty())
            return core::result<asset_search_property_filter, std::string>::failure("Property value is empty");
        filter.string_value = std::string(value);
        return core::result<asset_search_property_filter, std::string>::success(std::move(filter));
    }

    if (value.starts_with("<="))
    {
        filter.comparison = asset_search_comparison::less_equal;
        value.remove_prefix(2);
    }
    else if (value.starts_with(">="))
    {
        filter.comparison = asset_search_comparison::greater_equal;
        value.remove_prefix(2);
    }
    else if (value.starts_with("<"))
    {
        filter.comparison = asset_search_comparison::less;
        value.remove_prefix(1);
    }
    else if (value.starts_with(">"))
    {
        filter.comparison = asset_search_comparison::greater;
        value.remove_prefix(1);
    }
    else
    {
        filter.comparison = asset_search_comparison::equal;
        if (value.starts_with("=")) value.remove_prefix(1);
    }

    if (value.empty())
        return core::result<asset_search_property_filter, std::string>::failure("Numeric property value is empty");

    double number{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return core::result<asset_search_property_filter, std::string>::failure(
            "Numeric property value is invalid");

    filter.number_value = number;
    return core::result<asset_search_property_filter, std::string>::success(std::move(filter));
}

struct parsed_search_token
{
    std::string text;
    std::size_t offset{};
};

core::result<std::vector<parsed_search_token>, asset_search_parse_error> tokenize_search_query(std::string_view input)
{
    std::vector<parsed_search_token> tokens;
    std::size_t index{};
    while (index < input.size())
    {
        while (index < input.size() && std::isspace(static_cast<unsigned char>(input[index])))
            ++index;
        if (index >= input.size()) break;

        const auto start = index;
        std::string token;
        bool quoted{};
        while (index < input.size())
        {
            const auto character = input[index];
            if (character == '\\' && quoted && index + 1 < input.size() &&
                (input[index + 1] == '"' || input[index + 1] == '\\'))
            {
                token.push_back(input[index + 1]);
                index += 2;
                continue;
            }
            if (character == '"')
            {
                quoted = !quoted;
                ++index;
                continue;
            }
            if (!quoted && std::isspace(static_cast<unsigned char>(character))) break;
            token.push_back(character);
            ++index;
        }

        if (quoted)
            return core::result<std::vector<parsed_search_token>, asset_search_parse_error>::failure(
                {.offset = start, .token = token, .message = "Unterminated quoted search value"});

        if (!token.empty()) tokens.push_back({.text = std::move(token), .offset = start});
    }
    return core::result<std::vector<parsed_search_token>, asset_search_parse_error>::success(std::move(tokens));
}

} // namespace

asset_search_parse_result parse_asset_search_query(std::string_view text, std::size_t limit, std::size_t offset)
{
    auto tokenized = tokenize_search_query(text);
    if (!tokenized) return asset_search_parse_result::failure(std::move(tokenized).error());

    asset_search_query query;
    query.limit = limit;
    query.offset = offset;

    std::vector<std::string> free_text;
    for (const auto& token : tokenized.value())
    {
        const auto separator = token.text.find(':');
        if (separator == std::string::npos)
        {
            free_text.push_back(token.text);
            continue;
        }

        const auto key = lowercase_search_token(std::string_view(token.text).substr(0, separator));
        const auto value = std::string_view(token.text).substr(separator + 1);
        if (key.empty() || value.empty())
            return asset_search_parse_result::failure(
                {.offset = token.offset, .token = token.text, .message = "Search predicate requires key:value"});

        if (key == "type")
        {
            const auto type = search_type_from_alias(value);
            if (!type)
                return asset_search_parse_result::failure(
                    {.offset = token.offset, .token = token.text, .message = "Unknown asset type"});
            if (query.filters.type && *query.filters.type != *type)
                return asset_search_parse_result::failure(
                    {.offset = token.offset, .token = token.text, .message = "Only one type predicate is supported"});
            query.filters.type = *type;
        }
        else if (key == "tag")
        {
            append_unique(query.filters.required_tags, std::string(value));
        }
        else if (key == "category")
        {
            append_unique(query.filters.required_categories, std::string(value));
        }
        else if (key == "provider")
        {
            const auto normalized = lowercase_search_token(value);
            if (normalized == "project" || normalized == "builtin" || normalized == "built-in" ||
                normalized == "engine")
            {
                if (!apply_scope(value, query.filters.provider_ids))
                    return asset_search_parse_result::failure(
                        {.offset = token.offset, .token = token.text, .message = "Unknown provider alias"});
            }
            else
                append_unique(query.filters.provider_ids, std::string(value));
        }
        else if (key == "scope")
        {
            if (!apply_scope(value, query.filters.provider_ids))
                return asset_search_parse_result::failure(
                    {.offset = token.offset, .token = token.text, .message = "Unknown search scope"});
        }
        else if (const auto property_kind = search_property_kind(key))
        {
            auto property = parse_property_filter(key, value, *property_kind);
            if (!property)
                return asset_search_parse_result::failure(
                    {.offset = token.offset, .token = token.text, .message = std::move(property).error()});
            query.filters.property_filters.push_back(std::move(property).value());
        }
        else
        {
            return asset_search_parse_result::failure(
                {.offset = token.offset, .token = token.text, .message = "Unknown search predicate"});
        }
    }

    for (const auto& token : free_text)
    {
        if (!query.text.empty()) query.text.push_back(' ');
        query.text += token;
    }
    return asset_search_parse_result::success(std::move(query));
}

namespace
{

asset_status apply_changes(const asset_database& database, asset_search_index& index,
                           const asset_database_change_batch& batch)
{
    for (const auto& change : batch.events)
    {
        if (change.kind == asset_database_change_kind::catalog_reset || !change.guid.valid()) continue;

        asset_status status = asset_status::success();
        if (change.kind == asset_database_change_kind::removed)
            status = index.erase(change.guid);
        else if (const auto record = database.query(change.guid))
            status = index.upsert(make_asset_search_document(*record));
        else
            status = index.erase(change.guid);

        if (!status) return status;
    }
    return asset_status::success();
}

} // namespace

asset_search_document make_asset_search_document(const asset_database_record& record)
{
    asset_search_document document;
    document.guid = record.guid;
    document.type = record.type;
    document.display_name = record.title;
    document.logical_path = record.source_hint;
    document.active_provider = record.active_provider;
    document.title = record.title;
    document.description = record.description;
    document.authored_tags = record.authored_tags;
    document.categories = record.authored_categories;
    document.properties = {
        {.key = "dependencies",
         .kind = asset_search_property_kind::number,
         .number_value = static_cast<double>(record.dependencies.size())},
        {.key = "reverse-dependencies",
         .kind = asset_search_property_kind::number,
         .number_value = static_cast<double>(record.reverse_dependencies.size())},
        {.key = "generation",
         .kind = asset_search_property_kind::number,
         .number_value = static_cast<double>(record.generation)},
        {.key = "revision",
         .kind = asset_search_property_kind::number,
         .number_value = static_cast<double>(record.revision)}};
    document.source_generation = record.generation;
    document.source_revision = record.revision;
    return document;
}

asset_status asset_search_indexer::rebuild()
{
    if (!database_ || !index_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search indexer is not initialized"});

    const auto records = database_->records();
    std::vector<asset_search_document> documents;
    documents.reserve(records.size());
    for (const auto& record : records)
        documents.push_back(make_asset_search_document(record));

    auto status = index_->reset(documents);
    if (!status) return status;

    // Replaying retained events after the snapshot is intentionally idempotent.
    // It closes the race between taking the database snapshot and observing the
    // latest change cursor. history_lost is harmless here because the complete
    // record snapshot is already the baseline.
    const auto batch = database_->changes_since(0);
    status = apply_changes(*database_, *index_, batch);
    if (!status) return status;
    status = index_->set_source_change_cursor(batch.next_sequence);
    if (!status) return status;
    cursor_ = batch.next_sequence;
    return asset_status::success();
}

asset_status asset_search_indexer::refresh()
{
    if (!database_ || !index_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search indexer is not initialized"});

    const auto batch = database_->changes_since(cursor_);
    if (batch.history_lost ||
        std::any_of(batch.events.begin(), batch.events.end(), [](const asset_database_change& change)
                    { return change.kind == asset_database_change_kind::catalog_reset; }))
    {
        return rebuild();
    }

    auto status = apply_changes(*database_, *index_, batch);
    if (!status) return status;
    status = index_->set_source_change_cursor(batch.next_sequence);
    if (!status) return status;
    cursor_ = batch.next_sequence;
    return asset_status::success();
}

asset_search_service::asset_search_service(const asset_database& database,
                                           std::unique_ptr<persistent_asset_search_index> index)
    : database_(&database), index_(std::move(index)),
      indexer_(std::make_unique<asset_search_indexer>(database, *index_))
{
}

asset_search_service::~asset_search_service() = default;

asset_search_service::open_result asset_search_service::open(const asset_database& database,
                                                             std::filesystem::path cache_path)
{
    auto opened = persistent_asset_search_index::open(std::move(cache_path));
    if (!opened) return open_result::failure(std::move(opened).error());

    auto service = std::unique_ptr<asset_search_service>(new asset_search_service(database, std::move(opened).value()));
    auto status = service->synchronize();
    if (!status) return open_result::failure(std::move(status).error());
    return open_result::success(std::move(service));
}

asset_status asset_search_service::synchronize()
{
    std::scoped_lock lock(mutex_);
    if (!database_ || !index_ || !indexer_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search service is not initialized"});

    if (!index_->source_change_cursor()) return indexer_->rebuild();
    return indexer_->refresh();
}

asset_status asset_search_service::rebuild()
{
    std::scoped_lock lock(mutex_);
    if (!indexer_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search service is not initialized"});
    return indexer_->rebuild();
}

asset_status asset_search_service::refresh()
{
    std::scoped_lock lock(mutex_);
    if (!indexer_)
        return asset_status::failure(
            {.code = asset_error_code::invalid_request, .message = "Asset search service is not initialized"});
    return indexer_->refresh();
}

std::vector<asset_search_result> asset_search_service::query(const asset_search_query& query_value) const
{
    std::scoped_lock lock(mutex_);
    return index_ ? index_->query(query_value) : std::vector<asset_search_result>{};
}

std::uint64_t asset_search_service::cursor() const
{
    std::scoped_lock lock(mutex_);
    return indexer_ ? indexer_->cursor() : 0;
}

const std::filesystem::path& asset_search_service::path() const noexcept
{
    static const std::filesystem::path empty;
    return index_ ? index_->path() : empty;
}

} // namespace arc::assets
