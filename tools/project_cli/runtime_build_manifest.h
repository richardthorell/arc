#pragma once

#include <arc/project/package_workflow.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

struct runtime_build_inputs
{
    std::filesystem::path executable;
    std::vector<std::filesystem::path> libraries;
    std::vector<std::filesystem::path> plugins;
};

std::optional<runtime_build_inputs> load_runtime_build_inputs(const std::filesystem::path& path,
                                                              const arc::project::package_workflow_plan& plan,
                                                              std::string& error);
