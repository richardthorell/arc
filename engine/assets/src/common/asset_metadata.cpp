#include <arc/assets/assets.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <system_error>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>
#endif

namespace arc::assets
{
namespace
{

using json = nlohmann::json;

std::string lowercase(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

bool replace_file(const std::filesystem::path& source, const std::filesystem::path& destination) noexcept
{
#if defined(_WIN32)
    return MoveFileExW(
        source.c_str(),
        destination.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
#else
    std::error_code error;
    std::filesystem::rename(source, destination, error);
    return !error;
#endif
}

json subasset_json(const asset_subasset_metadata& subasset)
{
    return {
        { "key", subasset.persistent_key },
        { "guid", to_string(subasset.guid) },
        { "type", to_string(subasset.type) },
        { "name", subasset.name },
        { "tombstoned", subasset.tombstoned }
    };
}

}

std::filesystem::path metadata_path_for(const std::filesystem::path& source_path)
{
    return std::filesystem::path(source_path.native() +
        std::filesystem::path(".arcmeta").native());
}

bool load_asset_metadata(
    const std::filesystem::path& path,
    asset_source_metadata& metadata,
    std::string& error)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        error = "Could not open asset metadata";
        return false;
    }

    json document;
    try
    {
        input >> document;
    }
    catch (const std::exception& exception)
    {
        error = std::string("Invalid asset metadata JSON: ") + exception.what();
        return false;
    }

    if (!document.is_object() ||
        document.value("format", std::string{}) != "arc.asset-meta" ||
        document.value("formatVersion", 0u) != asset_source_metadata::current_format_version)
    {
        error = "Unsupported ARC asset metadata format or version";
        return false;
    }

    const auto guid = parse_asset_guid(document.value("guid", std::string{}));
    const auto type = parse_asset_type_id(document.value("type", std::string{}));
    const auto importer = parse_asset_importer_id(document.value("importer", std::string{}));
    if (!guid || !type || !importer)
    {
        error = "Asset metadata contains an invalid GUID, type, or importer ID";
        return false;
    }

    asset_source_metadata parsed;
    parsed.guid = *guid;
    parsed.type = *type;
    parsed.importer = *importer;
    parsed.settings_version = document.value("settingsVersion", 1u);
    if (parsed.settings_version == 0)
    {
        error = "Asset metadata settings version must be positive";
        return false;
    }
    const auto settings = document.find("settings");
    parsed.canonical_settings = settings == document.end() ? "{}" : settings->dump();

    if (const auto subassets = document.find("subassets"); subassets != document.end())
    {
        if (!subassets->is_array())
        {
            error = "Asset metadata subassets must be an array";
            return false;
        }
        for (const auto& record : *subassets)
        {
            if (!record.is_object())
            {
                error = "Asset metadata contains a malformed subasset";
                return false;
            }
            const auto subasset_guid = parse_asset_guid(record.value("guid", std::string{}));
            const auto subasset_type = parse_asset_type_id(record.value("type", std::string{}));
            const std::string key = record.value("key", std::string{});
            if (!subasset_guid || !subasset_type || key.empty())
            {
                error = "Asset metadata contains an invalid subasset";
                return false;
            }
            if (std::any_of(parsed.subassets.begin(), parsed.subassets.end(),
                    [&](const asset_subasset_metadata& value) {
                        return value.persistent_key == key || value.guid == *subasset_guid;
                    }))
            {
                error = "Asset metadata contains duplicate subasset keys or GUIDs";
                return false;
            }
            parsed.subassets.push_back({
                .persistent_key = key,
                .guid = *subasset_guid,
                .type = *subasset_type,
                .name = record.value("name", std::string{}),
                .tombstoned = record.value("tombstoned", false)
            });
        }
    }
    metadata = std::move(parsed);
    return true;
}

bool save_asset_metadata(
    const std::filesystem::path& path,
    const asset_source_metadata& metadata,
    std::string& error)
{
    if (!metadata.guid.valid() || !metadata.type.valid() || !metadata.importer.valid() ||
        metadata.settings_version == 0)
    {
        error = "Asset metadata is incomplete";
        return false;
    }

    json settings = json::parse(metadata.canonical_settings, nullptr, false);
    if (settings.is_discarded())
    {
        error = "Asset import settings are not valid JSON";
        return false;
    }

    json document{
        { "format", "arc.asset-meta" },
        { "formatVersion", asset_source_metadata::current_format_version },
        { "guid", to_string(metadata.guid) },
        { "type", to_string(metadata.type) },
        { "importer", to_string(metadata.importer) },
        { "settingsVersion", metadata.settings_version },
        { "settings", std::move(settings) },
        { "subassets", json::array() }
    };
    for (const asset_subasset_metadata& subasset : metadata.subassets)
        document["subassets"].push_back(subasset_json(subasset));

    std::error_code directory_error;
    std::filesystem::create_directories(path.parent_path(), directory_error);
    if (directory_error)
    {
        error = "Could not create the asset metadata directory";
        return false;
    }

    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto temporary = path.parent_path() /
        (path.filename().string() + ".tmp-" + std::to_string(stamp));
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            error = "Could not create temporary asset metadata";
            return false;
        }
        output << document.dump(2) << '\n';
        output.flush();
        if (!output)
        {
            error = "Could not flush asset metadata";
            std::filesystem::remove(temporary, directory_error);
            return false;
        }
    }
    if (!replace_file(temporary, path))
    {
        std::filesystem::remove(temporary, directory_error);
        error = "Could not atomically replace asset metadata";
        return false;
    }
    return true;
}

std::string normalize_asset_path(const std::filesystem::path& path)
{
    return path.lexically_normal().generic_string();
}

std::optional<std::pair<asset_type_id, asset_importer_id>> classify_asset_path(
    const std::filesystem::path& path) noexcept
{
    const std::string extension = lowercase(path.extension().string());
    if (extension == ".arcscene")
        return std::pair{ asset_types::scene, importer_ids::scene };
    if (extension == ".arcprefab")
        return std::pair{ asset_types::prefab, importer_ids::prefab };
    if (extension == ".arcmat")
        return std::pair{ asset_types::material, importer_ids::material };
    if (extension == ".slang" || extension == ".glsl" || extension == ".vert" ||
        extension == ".frag" || extension == ".comp" || extension == ".hlsl" ||
        extension == ".inc")
        return std::pair{ asset_types::shader, importer_ids::shader };
    if (extension == ".hdr" || extension == ".exr")
        return std::pair{ asset_types::environment, importer_ids::environment };
    if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" ||
        extension == ".dds" || extension == ".tga" || extension == ".bmp" ||
        extension == ".ktx" || extension == ".ktx2")
        return std::pair{ asset_types::texture_2d, importer_ids::texture };
    if (extension == ".glb" || extension == ".gltf")
        return std::pair{ asset_types::imported_scene, importer_ids::gltf };
    if (extension == ".fbx")
        return std::pair{ asset_types::imported_scene, importer_ids::fbx };
    if (extension == ".bin")
        return std::pair{ asset_types::binary_blob, importer_ids::binary };
    if (extension == ".arcanim")
        return std::pair{ asset_types::animation_clip, importer_ids::animation };
    if (extension == ".arccollision")
        return std::pair{ asset_types::collision, importer_ids::collision };
    if (extension == ".arcnav")
        return std::pair{ asset_types::navigation, importer_ids::navigation };
    if (extension == ".wav" || extension == ".ogg" || extension == ".mp3" ||
        extension == ".flac")
        return std::pair{ asset_types::audio_clip, importer_ids::audio };
    return std::nullopt;
}

} // namespace arc::assets
