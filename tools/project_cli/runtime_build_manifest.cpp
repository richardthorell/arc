#include "runtime_build_manifest.h"

#include <arc/assets/assets.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string_view>

std::optional<runtime_build_inputs> load_runtime_build_inputs(const std::filesystem::path& path,
                                                              const arc::project::package_workflow_plan& plan,
                                                              std::string& error)
{
    namespace fs = std::filesystem;
    using json = nlohmann::json;
    try
    {
        if (fs::file_size(path) > 1024 * 1024) throw std::runtime_error("Build manifest exceeds 1 MiB");
        std::ifstream stream(path);
        const auto document = json::parse(stream);
        if (document.at("format") != "arc.runtime-build" || document.at("formatVersion") != 1)
            throw std::runtime_error("Unsupported runtime build manifest format/version");
        if (document.at("projectGuid") != plan.project_guid || document.at("engineVersion") != plan.engine_version ||
            document.at("profile") != plan.profile || document.at("targetId") != plan.target_id ||
            document.at("configuration") != plan.configuration ||
            document.at("role") != (plan.role == arc::project::package_workflow_role::server ? "server" : "client"))
            throw std::runtime_error("Runtime build identity does not match the selected project/profile/target");
        if (document.at("cmakeTarget").get<std::string>().empty())
            throw std::runtime_error("Runtime build target is missing");
        std::set<fs::path> seen;
        const auto file = [&](const json& record)
        {
            const fs::path source(record.at("path").get<std::string>());
            if (!source.is_absolute() || !fs::is_regular_file(source))
                throw std::runtime_error("Runtime build input must be an existing absolute file: " + source.string());
            const auto canonical = fs::canonical(source);
            if (!seen.insert(canonical).second) throw std::runtime_error("Duplicate runtime build input");
            const auto hash = arc::assets::hash_file(canonical);
            if (!record.at("size").is_number_unsigned() ||
                fs::file_size(canonical) != record.at("size").get<std::uintmax_t>() || !hash ||
                arc::assets::to_string(hash.value()) != record.at("sha256").get<std::string>())
                throw std::runtime_error("Runtime build input changed since compilation: " + canonical.string());
            return canonical;
        };
        runtime_build_inputs result;
        result.executable = file(document.at("executable"));
        for (const auto* field : {"libraries", "plugins"})
        {
            const auto& records = document.at(field);
            if (!records.is_array() || records.size() > 256)
                throw std::runtime_error("Runtime dependency inventory exceeds bounds or is not an array");
            auto& output = std::string_view(field) == "libraries" ? result.libraries : result.plugins;
            for (const auto& record : records)
                output.push_back(file(record));
        }
        return result;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return std::nullopt;
    }
}
