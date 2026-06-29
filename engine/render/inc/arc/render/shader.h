#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
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
    std::vector<std::string> defines;
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

/**
 * @brief Cached shader source metadata used for hot reload checks.
 */
struct shader_source_state
{
    std::filesystem::path path;
    std::filesystem::file_time_type last_write_time{};
    std::uintmax_t size{};

    friend bool operator==(const shader_source_state& lhs, const shader_source_state& rhs) noexcept
    {
        return lhs.path == rhs.path && lhs.last_write_time == rhs.last_write_time && lhs.size == rhs.size;
    }
};

/**
 * @brief Backend-neutral shader compiler interface.
 */
class shader_compiler
{
public:
    virtual ~shader_compiler() = default;

    /**
     * @brief Compile one shader request into backend bytecode.
     */
    virtual shader_compile_result compile(const shader_compile_request& request) = 0;
};

/**
 * @brief Shader library with request caching and source timestamp tracking.
 */
class shader_library_cache
{
public:
    /**
     * @brief Compile or return a cached result for the request.
     */
    shader_compile_result compile_or_get(shader_compiler& compiler, const shader_compile_request& request);

    /**
     * @brief Return whether the source file for a request has changed since caching.
     */
    bool source_changed(const shader_compile_request& request) const;

    /**
     * @brief Remove all cached shader results.
     */
    void clear();

    /**
     * @brief Return number of cached shader requests.
     */
    std::size_t size() const noexcept;

private:
    struct cached_shader
    {
        shader_compile_result result;
        std::optional<shader_source_state> source;
    };

    std::unordered_map<std::string, cached_shader> cache_;
};

} // namespace arc::render
