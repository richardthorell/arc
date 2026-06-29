#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace arc::render
{

/**
 * @brief Backend shader output format.
 */
enum class shader_target : std::uint8_t
{
    spirv,
    dxil,
    msl
};

/**
 * @brief Shader compilation input.
 */
struct shader_compile_request
{
    std::string source_path;
    std::string entry_point{ "main" };
    std::string profile;
    shader_target target{ shader_target::spirv };
};

/**
 * @brief Minimal shader reflection data.
 */
struct shader_reflection
{
    std::vector<std::string> entry_points;
    std::vector<std::string> resources;
};

/**
 * @brief Shader compilation result.
 */
struct shader_compile_result
{
    bool succeeded{};
    std::string diagnostics;
    std::vector<std::uint8_t> bytecode;
    shader_reflection reflection;
};

} // namespace arc::render
