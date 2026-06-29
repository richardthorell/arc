#include <arc/render/shader.h>

#include <sstream>

namespace arc::render
{
namespace
{

std::string shader_cache_key(const shader_compile_request& request)
{
    std::ostringstream key;
    key << request.source_path << '|'
        << request.entry_point << '|'
        << request.profile << '|'
        << static_cast<int>(request.target);
    for (const auto& define : request.defines)
        key << '|' << define;
    return key.str();
}

std::optional<shader_source_state> read_source_state(const std::filesystem::path& path)
{
    std::error_code error;
    if (!std::filesystem::exists(path, error) || error)
        return std::nullopt;

    const auto write_time = std::filesystem::last_write_time(path, error);
    if (error)
        return std::nullopt;

    const auto size = std::filesystem::file_size(path, error);
    if (error)
        return std::nullopt;

    return shader_source_state{ .path = path, .last_write_time = write_time, .size = size };
}

} // namespace

shader_compile_result shader_library_cache::compile_or_get(shader_compiler& compiler, const shader_compile_request& request)
{
    const auto key = shader_cache_key(request);
    const auto current_source = read_source_state(request.source_path);

    if (const auto found = cache_.find(key); found != cache_.end())
    {
        if (!current_source || found->second.source == current_source)
            return found->second.result;
    }

    auto result = compiler.compile(request);
    if (result.succeeded)
        cache_[key] = { .result = result, .source = current_source };
    return result;
}

bool shader_library_cache::source_changed(const shader_compile_request& request) const
{
    const auto found = cache_.find(shader_cache_key(request));
    if (found == cache_.end())
        return true;

    const auto current_source = read_source_state(request.source_path);
    return found->second.source != current_source;
}

void shader_library_cache::clear()
{
    cache_.clear();
}

std::size_t shader_library_cache::size() const noexcept
{
    return cache_.size();
}

} // namespace arc::render
