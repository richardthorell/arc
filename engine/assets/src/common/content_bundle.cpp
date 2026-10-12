#include <arc/assets/content_bundle.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>
#include <system_error>

namespace arc::assets
{
namespace
{
using json = nlohmann::json;

bool valid_descriptor(const content_bundle_descriptor& descriptor)
{
    if (descriptor.version != content_bundle_descriptor::current_version || descriptor.id.empty()) return false;
    for (const auto guid : descriptor.roots)
        if (!guid.valid()) return false;
    for (const auto& dependency : descriptor.bundle_dependencies)
        if (dependency.empty() || dependency == descriptor.id) return false;
    return true;
}

template <typename T> bool read_string_array(const json& document, const char* key, std::vector<T>& result)
{
    if (!document.contains(key)) return true;
    if (!document[key].is_array()) return false;
    for (const auto& value : document[key])
    {
        if (!value.is_string()) return false;
        result.emplace_back(value.get<std::string>());
    }
    return true;
}
} // namespace

content_bundle_io_result parse_content_bundle(std::string_view text)
{
    content_bundle_io_result result;
    const auto document = json::parse(text.begin(), text.end(), nullptr, false);
    if (!document.is_object() || !document.contains("format") || !document["format"].is_string() ||
        document["format"] != "arc.bundle" || !document.contains("version") ||
        !document["version"].is_number_unsigned() ||
        document["version"].get<std::uint64_t>() > content_bundle_descriptor::current_version ||
        !document.contains("id") || !document["id"].is_string() ||
        (document.contains("preload") && !document["preload"].is_string()) ||
        (document.contains("closure") && !document["closure"].is_string()))
    {
        result.error = "Invalid .arcbundle format, version or ID";
        return result;
    }

    auto& descriptor = result.descriptor;
    descriptor.version = document["version"].get<std::uint32_t>();
    descriptor.id = document["id"].get<std::string>();
    if (document.contains("displayName"))
    {
        if (!document["displayName"].is_string())
        {
            result.error = "Invalid displayName";
            return result;
        }
        descriptor.display_name = document["displayName"].get<std::string>();
    }
    if (!document.contains("roots") || !document["roots"].is_array())
    {
        result.error = "Bundle roots must be an array";
        return result;
    }
    for (const auto& value : document["roots"])
    {
        const auto parsed = value.is_string() ? parse_asset_guid(value.get<std::string>()) : std::nullopt;
        if (!parsed)
        {
            result.error = "Invalid root asset GUID";
            return result;
        }
        descriptor.roots.push_back(*parsed);
    }
    if (!read_string_array(document, "bundleDependencies", descriptor.bundle_dependencies) ||
        !read_string_array(document, "tags", descriptor.tags))
    {
        result.error = "Bundle dependencies and tags must contain only strings";
        return result;
    }

    const auto preload = document.value("preload", std::string{"none"});
    if (preload == "none")
        descriptor.preload = bundle_preload_policy::none;
    else if (preload == "metadata")
        descriptor.preload = bundle_preload_policy::metadata;
    else if (preload == "eager")
        descriptor.preload = bundle_preload_policy::eager;
    else
    {
        result.error = "Unknown bundle preload policy";
        return result;
    }

    const auto closure = document.value("closure", std::string{"transitive"});
    if (closure == "transitive")
        descriptor.closure = bundle_closure_policy::transitive_dependencies;
    else if (closure == "roots-only")
        descriptor.closure = bundle_closure_policy::roots_only;
    else
    {
        result.error = "Unknown bundle closure policy";
        return result;
    }

    if (!valid_descriptor(descriptor)) result.error = "Invalid bundle descriptor or unsupported version";
    return result;
}

content_bundle_io_result load_content_bundle(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) return {.error = "Could not open .arcbundle file"};
    const std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return parse_content_bundle(bytes);
}

std::string serialize_content_bundle(const content_bundle_descriptor& descriptor)
{
    if (!valid_descriptor(descriptor)) return {};
    json roots = json::array();
    for (const auto guid : descriptor.roots)
        roots.push_back(to_string(guid));
    const json document{
        {"format", "arc.bundle"},
        {"version", descriptor.version},
        {"id", descriptor.id},
        {"displayName", descriptor.display_name},
        {"roots", std::move(roots)},
        {"bundleDependencies", descriptor.bundle_dependencies},
        {"preload", descriptor.preload == bundle_preload_policy::eager      ? "eager"
                    : descriptor.preload == bundle_preload_policy::metadata ? "metadata"
                                                                            : "none"},
        {"closure", descriptor.closure == bundle_closure_policy::roots_only ? "roots-only" : "transitive"},
        {"tags", descriptor.tags}};
    return document.dump(2) + "\n";
}

content_bundle_save_result save_content_bundle(const std::filesystem::path& path,
                                               const content_bundle_descriptor& descriptor)
{
    const auto bytes = serialize_content_bundle(descriptor);
    if (bytes.empty()) return {.error = "Invalid bundle descriptor"};
    std::error_code ec;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return {.error = ec.message()};
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file || !(file << bytes)) return {.error = "Could not write .arcbundle file"};
    return {};
}
} // namespace arc::assets
