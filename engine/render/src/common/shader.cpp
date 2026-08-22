#include <arc/render/shader.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <mutex>
#include <limits>
#include <set>
#include <sstream>
#include <type_traits>

namespace arc::render
{
namespace
{

constexpr std::string_view package_magic = "ARC_SHADER_2";
constexpr std::size_t maximum_package_bytes = 256u * 1024u * 1024u;
constexpr std::size_t maximum_package_entries = 65'536u;

constexpr std::array<std::uint32_t, 64> sha256_constants{
    {0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
     0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
     0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
     0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
     0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
     0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
     0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
     0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u}};

class sha256
{
public:
    void update(std::span<const std::byte> bytes) noexcept
    {
        for (const std::byte value : bytes)
        {
            buffer_[buffer_size_++] = std::to_integer<std::uint8_t>(value);
            if (buffer_size_ == buffer_.size())
            {
                transform();
                bit_count_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    void update(std::string_view text) noexcept
    {
        update(std::as_bytes(std::span(text.data(), text.size())));
    }

    shader_content_hash finish() noexcept
    {
        bit_count_ += static_cast<std::uint64_t>(buffer_size_) * 8u;
        buffer_[buffer_size_++] = 0x80u;
        if (buffer_size_ > 56)
        {
            while (buffer_size_ < 64)
                buffer_[buffer_size_++] = 0;
            transform();
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56)
            buffer_[buffer_size_++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8)
            buffer_[buffer_size_++] = static_cast<std::uint8_t>(bit_count_ >> shift);
        transform();

        shader_content_hash result;
        for (std::size_t word = 0; word < state_.size(); ++word)
            for (std::size_t byte = 0; byte < 4; ++byte)
                result.bytes[word * 4 + byte] = static_cast<std::byte>(state_[word] >> ((3 - byte) * 8));
        return result;
    }

private:
    void transform() noexcept
    {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index)
            words[index] = (static_cast<std::uint32_t>(buffer_[index * 4]) << 24) |
                           (static_cast<std::uint32_t>(buffer_[index * 4 + 1]) << 16) |
                           (static_cast<std::uint32_t>(buffer_[index * 4 + 2]) << 8) |
                           static_cast<std::uint32_t>(buffer_[index * 4 + 3]);
        for (std::size_t index = 16; index < words.size(); ++index)
        {
            const auto s0 =
                std::rotr(words[index - 15], 7) ^ std::rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const auto s1 =
                std::rotr(words[index - 2], 17) ^ std::rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        auto [a, b, c, d, e, f, g, h] = state_;
        for (std::size_t index = 0; index < words.size(); ++index)
        {
            const auto sum1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto temporary1 = h + sum1 + choice + sha256_constants[index] + words[index];
            const auto sum0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u}};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_{};
    std::uint64_t bit_count_{};
};

shader_content_hash hash_text(std::string_view text) noexcept
{
    sha256 hash;
    hash.update(text);
    return hash.finish();
}

std::optional<std::string> read_text(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return std::nullopt;
    std::ostringstream text;
    text << input.rdbuf();
    if (!input.eof() && input.fail()) return std::nullopt;
    return text.str();
}

std::optional<std::filesystem::path> resolve_include(const std::filesystem::path& including_file,
                                                     std::string_view include,
                                                     const shader_compile_request& request)
{
    std::error_code error;
    auto candidate = (including_file.parent_path() / include).lexically_normal();
    if (std::filesystem::is_regular_file(candidate, error) && !error) return candidate;
    for (const auto& directory : request.include_directories)
    {
        error.clear();
        candidate = (directory / include).lexically_normal();
        if (std::filesystem::is_regular_file(candidate, error) && !error) return candidate;
    }
    return std::nullopt;
}

std::vector<std::string> includes_in(std::string_view source)
{
    std::vector<std::string> result;
    std::size_t cursor{};
    while (cursor < source.size())
    {
        const auto line_end = source.find('\n', cursor);
        const auto line = source.substr(cursor, line_end == std::string_view::npos ? source.size() - cursor
                                                                                  : line_end - cursor);
        const auto directive = line.find("#include");
        if (directive != std::string_view::npos)
        {
            const auto begin = line.find_first_of("\"<", directive + 8);
            if (begin != std::string_view::npos)
            {
                const char close = line[begin] == '<' ? '>' : '\"';
                const auto end = line.find(close, begin + 1);
                if (end != std::string_view::npos) result.emplace_back(line.substr(begin + 1, end - begin - 1));
            }
        }
        if (line_end == std::string_view::npos) break;
        cursor = line_end + 1;
    }
    return result;
}

std::vector<std::string> imports_in(std::string_view source)
{
    std::vector<std::string> result;
    std::size_t cursor{};
    while (cursor < source.size())
    {
        const auto line_end = source.find('\n', cursor);
        auto line = source.substr(cursor, line_end == std::string_view::npos ? source.size() - cursor
                                                                            : line_end - cursor);
        const auto first = line.find_first_not_of(" \t");
        if (first != std::string_view::npos && line.substr(first).starts_with("import "))
        {
            const auto begin = line.find_first_not_of(" \t", first + 7);
            const auto end = begin == std::string_view::npos ? begin : line.find_first_of("; \t\r", begin);
            if (begin != std::string_view::npos && end != begin)
                result.emplace_back(line.substr(begin, end == std::string_view::npos ? line.size() - begin
                                                                                     : end - begin));
        }
        if (line_end == std::string_view::npos) break;
        cursor = line_end + 1;
    }
    return result;
}

std::optional<std::filesystem::path> resolve_import(const std::filesystem::path& importing_file,
                                                     std::string_view module,
                                                     const shader_compile_request& request)
{
    std::array<std::string, 3> names{std::string(module), std::string(module), std::string(module)};
    std::ranges::replace(names[1], '.', '/');
    std::ranges::replace(names[2], '.', '_');
    for (auto& name : names)
        name += ".slang";
    std::error_code error;
    const auto probe = [&](const std::filesystem::path& directory) -> std::optional<std::filesystem::path>
    {
        for (const auto& name : names)
        {
            error.clear();
            auto candidate = (directory / name).lexically_normal();
            if (std::filesystem::is_regular_file(candidate, error) && !error) return candidate;
        }
        return std::nullopt;
    };
    if (auto path = probe(importing_file.parent_path())) return path;
    for (const auto& directory : request.include_directories)
        if (auto path = probe(directory)) return path;
    return std::nullopt;
}

core::result<std::vector<shader_dependency>, shader_compile_error>
collect_dependencies(const shader_compile_request& request)
{
    if (request.source_path.empty())
        return core::result<std::vector<shader_dependency>, shader_compile_error>::failure(
            {.code = shader_compile_error_code::invalid_request, .message = "shader source path is empty"});

    std::vector<shader_dependency> dependencies;
    std::set<std::filesystem::path> visited;
    std::set<std::filesystem::path> active;
    const std::filesystem::path root = std::filesystem::path(request.source_path).lexically_normal();

    const auto visit = [&](const auto& self, const std::filesystem::path& path,
                           const std::optional<std::string>& override_source)
        -> std::optional<shader_compile_error>
    {
        const auto normalized = path.lexically_normal();
        if (active.contains(normalized))
            return shader_compile_error{.code = shader_compile_error_code::dependency_cycle,
                                        .source_path = normalized.generic_string(),
                                        .message = "shader include dependency cycle detected"};
        if (visited.contains(normalized)) return std::nullopt;

        const auto source = override_source ? override_source : read_text(normalized);
        if (!source)
            return shader_compile_error{.code = shader_compile_error_code::source_unavailable,
                                        .source_path = normalized.generic_string(),
                                        .message = "shader source or include could not be read"};

        active.insert(normalized);
        for (const auto& include : includes_in(*source))
        {
            const auto resolved = resolve_include(normalized, include, request);
            if (!resolved)
                return shader_compile_error{.code = shader_compile_error_code::source_unavailable,
                                            .source_path = normalized.generic_string(),
                                            .message = "shader include could not be resolved: " + include};
            if (auto error = self(self, *resolved, std::nullopt)) return error;
        }
        for (const auto& imported : imports_in(*source))
        {
            const auto resolved = resolve_import(normalized, imported, request);
            if (!resolved)
                return shader_compile_error{.code = shader_compile_error_code::source_unavailable,
                                            .source_path = normalized.generic_string(),
                                            .message = "shader module import could not be resolved: " + imported};
            if (auto error = self(self, *resolved, std::nullopt)) return error;
        }
        active.erase(normalized);
        visited.insert(normalized);
        dependencies.push_back(
            {.path = normalized.generic_string(), .content_hash = hash_text(*source)});
        return std::nullopt;
    };

    const std::optional<std::string> source_override =
        request.source_override.empty() ? std::nullopt : std::optional<std::string>{request.source_override};
    if (auto error = visit(visit, root, source_override))
        return core::result<std::vector<shader_dependency>, shader_compile_error>::failure(std::move(*error));

    std::ranges::sort(dependencies, {}, &shader_dependency::path);
    return core::result<std::vector<shader_dependency>, shader_compile_error>::success(std::move(dependencies));
}

shader_content_hash request_hash(const shader_compile_request& request,
                                 std::span<const shader_dependency> dependencies,
                                 std::string_view compiler_fingerprint) noexcept
{
    sha256 hash;
    const auto append_number = [&](auto value)
    { hash.update(std::as_bytes(std::span(&value, 1))); };
    hash.update("ARC_SHADER_BUILD_2");
    hash.update(request.source_path);
    hash.update(request.entry_point);
    hash.update(request.profile);
    hash.update(request.library_version);
    append_number(request.domain);
    append_number(request.stage);
    append_number(request.target);
    append_number(request.optimization);
    hash.update(compiler_fingerprint);
    for (const auto& define : request.defines)
        hash.update(define);
    for (const auto& value : request.static_switches)
    {
        append_number(value.id.value);
        hash.update(value.name);
        append_number(value.value);
    }
    for (const auto& dependency : dependencies)
    {
        hash.update(dependency.path);
        hash.update(dependency.content_hash.bytes);
    }
    return hash.finish();
}

std::string shader_cache_key(const shader_compile_request& request)
{
    std::ostringstream key;
    key << request.source_path << '|' << request.entry_point << '|' << request.profile << '|'
        << request.library_version << '|'
        << static_cast<int>(request.domain) << '|' << static_cast<int>(request.stage) << '|'
        << static_cast<int>(request.target);
    for (const auto& define : request.defines)
        key << '|' << define;
    for (const auto& value : request.static_switches)
        key << '|' << value.id.value << '=' << value.value;
    return key.str();
}

class binary_writer
{
public:
    template <class T> void value(T value)
        requires(std::is_integral_v<T> || std::is_enum_v<T>)
    {
        if constexpr (std::is_enum_v<T>)
        {
            using value_type = std::underlying_type_t<T>;
            this->value(static_cast<value_type>(value));
        }
        else
        {
            using value_type = T;
            for (std::size_t index = 0; index < sizeof(value_type); ++index)
                bytes.push_back(static_cast<std::byte>((value >> (index * 8u)) & static_cast<value_type>(0xff)));
        }
    }

    void string(std::string_view text)
    {
        value<std::uint32_t>(static_cast<std::uint32_t>(text.size()));
        bytes.insert(bytes.end(), reinterpret_cast<const std::byte*>(text.data()),
                     reinterpret_cast<const std::byte*>(text.data() + text.size()));
    }

    void raw(std::span<const std::byte> source)
    {
        value<std::uint32_t>(static_cast<std::uint32_t>(source.size()));
        bytes.insert(bytes.end(), source.begin(), source.end());
    }

    std::vector<std::byte> bytes;
};

class binary_reader
{
public:
    explicit binary_reader(std::span<const std::byte> bytes) : bytes_(bytes) {}

    template <class T> bool value(T& output)
        requires(std::is_integral_v<T> || std::is_enum_v<T>)
    {
        if constexpr (std::is_enum_v<T>)
        {
            std::underlying_type_t<T> converted{};
            if (!value(converted)) return false;
            output = static_cast<T>(converted);
            return true;
        }
        else
        {
            if (remaining() < sizeof(T)) return false;
            T converted{};
            for (std::size_t index = 0; index < sizeof(T); ++index)
                converted |= static_cast<T>(std::to_integer<unsigned char>(bytes_[cursor_ + index]))
                             << (index * 8u);
            cursor_ += sizeof(T);
            output = converted;
            return true;
        }
    }

    bool string(std::string& output)
    {
        std::uint32_t size{};
        if (!value(size) || size > remaining()) return false;
        output.assign(reinterpret_cast<const char*>(bytes_.data() + cursor_), size);
        cursor_ += size;
        return true;
    }

    bool raw(std::vector<std::uint8_t>& output)
    {
        std::uint32_t size{};
        if (!value(size) || size > remaining()) return false;
        output.resize(size);
        if (size != 0) std::memcpy(output.data(), bytes_.data() + cursor_, size);
        cursor_ += size;
        return true;
    }

    bool hash(shader_content_hash& output)
    {
        if (remaining() < output.bytes.size()) return false;
        std::copy_n(bytes_.begin() + static_cast<std::ptrdiff_t>(cursor_), output.bytes.size(), output.bytes.begin());
        cursor_ += output.bytes.size();
        return true;
    }

    [[nodiscard]] std::size_t remaining() const noexcept { return bytes_.size() - cursor_; }

private:
    std::span<const std::byte> bytes_;
    std::size_t cursor_{};
};

shader_compile_error corrupt_package(std::string message)
{
    return {.code = shader_compile_error_code::package_corrupt, .message = std::move(message)};
}

} // namespace

shader_parameter_id make_shader_parameter_id(std::string_view stable_name) noexcept
{
    std::uint64_t hash = 14695981039346656037ull;
    for (const unsigned char character : stable_name)
    {
        hash ^= character;
        hash *= 1099511628211ull;
    }
    if (hash == 0) hash = 1;
    return shader_parameter_id{hash};
}

shader_entry_point_id make_shader_entry_point_id(std::string_view stable_name, shader_stage stage) noexcept
{
    std::uint64_t hash = make_shader_parameter_id(stable_name).value;
    hash ^= static_cast<std::uint64_t>(stage) + 0x9e3779b97f4a7c15ull + (hash << 6u) + (hash >> 2u);
    if (hash == 0) hash = 1;
    return shader_entry_point_id{hash};
}

std::string to_string(const shader_content_hash& hash)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(hash.bytes.size() * 2);
    for (const std::byte byte : hash.bytes)
    {
        const auto value = std::to_integer<unsigned>(byte);
        result.push_back(digits[value >> 4u]);
        result.push_back(digits[value & 0x0fu]);
    }
    return result;
}

shader_package_bytes_result serialize_shader_package(const shader_package& package)
{
    if (package.version != shader_package::current_version || !package.id.valid() || !package.generation.valid() ||
        !package.permutation.valid() || package.compiled.bytecode.empty() || package.compiled.build_hash.empty())
        return shader_package_bytes_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .message = "shader package identity, generation, permutation, bytecode, or build hash is invalid"});
    if (package.compiled.dependencies.size() > maximum_package_entries ||
        package.compiled.reflection.entry_points.size() > maximum_package_entries ||
        package.compiled.reflection.parameters.size() > maximum_package_entries ||
        package.compiled.reflection.resources.size() > maximum_package_entries ||
        package.compiled.reflection.passes.size() > maximum_package_entries ||
        package.compiled.source_map.size() > maximum_package_entries ||
        package.compiled.diagnostics.size() > maximum_package_entries)
        return shader_package_bytes_result::failure(
            {.code = shader_compile_error_code::validation_failed, .message = "shader package table exceeds limits"});
    if (std::ranges::any_of(package.compiled.diagnostics, [](const shader_diagnostic& diagnostic)
                           { return diagnostic.include_stack.size() > maximum_package_entries; }))
        return shader_package_bytes_result::failure(
            {.code = shader_compile_error_code::validation_failed,
             .message = "shader diagnostic include stack exceeds limits"});

    binary_writer writer;
    writer.string(package_magic);
    writer.value(package.version);
    writer.value(package.id.high);
    writer.value(package.id.low);
    writer.value(package.generation.value);
    writer.value(package.target);
    writer.value(package.permutation.value);
    writer.bytes.insert(writer.bytes.end(), package.compiled.build_hash.bytes.begin(), package.compiled.build_hash.bytes.end());
    writer.string(package.compiled.compiler_fingerprint);
    writer.raw(std::as_bytes(std::span(package.compiled.bytecode)));

    const auto& reflection = package.compiled.reflection;
    writer.value(reflection.domain);
    writer.value(reflection.parameter_block_size);
    writer.value<std::uint8_t>(reflection.custom_lighting ? 1 : 0);
    writer.value<std::uint8_t>(reflection.vertex_deformation ? 1 : 0);
    writer.value<std::uint8_t>(reflection.previous_vertex_deformation ? 1 : 0);

    writer.value<std::uint32_t>(static_cast<std::uint32_t>(reflection.entry_points.size()));
    for (const auto& entry : reflection.entry_points)
    {
        writer.value(entry.id.value);
        writer.string(entry.name);
        writer.value(entry.stage);
        writer.string(entry.profile);
        for (const auto size : entry.thread_group_size)
            writer.value(size);
    }
    writer.value<std::uint32_t>(static_cast<std::uint32_t>(reflection.parameters.size()));
    for (const auto& parameter : reflection.parameters)
    {
        writer.value(parameter.id.value);
        writer.string(parameter.name);
        writer.value(parameter.type);
        writer.value(parameter.offset);
        writer.value(parameter.size);
        writer.raw(parameter.default_value);
    }
    writer.value<std::uint32_t>(static_cast<std::uint32_t>(reflection.resources.size()));
    for (const auto& resource : reflection.resources)
    {
        writer.string(resource.name);
        writer.value(resource.kind);
        writer.value(resource.set);
        writer.value(resource.binding);
        writer.value(resource.count);
        writer.value<std::uint8_t>(resource.writable ? 1 : 0);
    }
    writer.value<std::uint32_t>(static_cast<std::uint32_t>(reflection.passes.size()));
    for (const auto& pass : reflection.passes)
    {
        writer.value(pass.pass);
        writer.value(pass.entry_point.value);
        writer.value<std::uint8_t>(pass.generated ? 1 : 0);
    }
    writer.value<std::uint32_t>(static_cast<std::uint32_t>(package.compiled.dependencies.size()));
    for (const auto& dependency : package.compiled.dependencies)
    {
        writer.string(dependency.path);
        writer.bytes.insert(writer.bytes.end(), dependency.content_hash.bytes.begin(), dependency.content_hash.bytes.end());
    }
    const auto write_location = [&](const shader_source_location& location)
    {
        writer.string(location.path);
        writer.value(location.line);
        writer.value(location.column);
        writer.string(location.graph_node_id);
    };
    writer.value<std::uint32_t>(static_cast<std::uint32_t>(package.compiled.source_map.size()));
    for (const auto& mapping : package.compiled.source_map)
    {
        writer.value(mapping.generated_line);
        write_location(mapping.source);
    }
    writer.value<std::uint32_t>(static_cast<std::uint32_t>(package.compiled.diagnostics.size()));
    for (const auto& diagnostic : package.compiled.diagnostics)
    {
        writer.value(diagnostic.severity);
        writer.string(diagnostic.code);
        writer.string(diagnostic.message);
        write_location(diagnostic.location);
        writer.value<std::uint32_t>(static_cast<std::uint32_t>(diagnostic.include_stack.size()));
        for (const auto& location : diagnostic.include_stack)
            write_location(location);
        writer.value<std::uint8_t>(diagnostic.permutation.has_value() ? 1 : 0);
        if (diagnostic.permutation) writer.value(diagnostic.permutation->representation());
    }

    if (writer.bytes.size() > maximum_package_bytes)
        return shader_package_bytes_result::failure(
            {.code = shader_compile_error_code::validation_failed, .message = "shader package exceeds size limit"});
    return shader_package_bytes_result::success(std::move(writer.bytes));
}

shader_package_result deserialize_shader_package(std::span<const std::byte> bytes)
{
    if (bytes.size() > maximum_package_bytes)
        return shader_package_result::failure(corrupt_package("shader package exceeds size limit"));
    binary_reader reader(bytes);
    shader_package package;
    std::string magic;
    if (!reader.string(magic) || magic != package_magic || !reader.value(package.version) ||
        package.version != shader_package::current_version || !reader.value(package.id.high) ||
        !reader.value(package.id.low) || !reader.value(package.generation.value) || !reader.value(package.target) ||
        !reader.value(package.permutation.value) || !reader.hash(package.compiled.build_hash) ||
        !reader.string(package.compiled.compiler_fingerprint) || !reader.raw(package.compiled.bytecode))
        return shader_package_result::failure(corrupt_package("shader package header is invalid or truncated"));

    auto& reflection = package.compiled.reflection;
    std::uint8_t custom_lighting{};
    std::uint8_t vertex_deformation{};
    std::uint8_t previous_deformation{};
    if (!reader.value(reflection.domain) || !reader.value(reflection.parameter_block_size) ||
        !reader.value(custom_lighting) || !reader.value(vertex_deformation) || !reader.value(previous_deformation))
        return shader_package_result::failure(corrupt_package("shader reflection header is truncated"));
    reflection.custom_lighting = custom_lighting != 0;
    reflection.vertex_deformation = vertex_deformation != 0;
    reflection.previous_vertex_deformation = previous_deformation != 0;

    const auto read_count = [&](std::uint32_t& count)
    { return reader.value(count) && count <= maximum_package_entries; };
    std::uint32_t count{};
    if (!read_count(count)) return shader_package_result::failure(corrupt_package("invalid entry-point count"));
    reflection.entry_points.resize(count);
    for (auto& entry : reflection.entry_points)
    {
        if (!reader.value(entry.id.value) || !reader.string(entry.name) || !reader.value(entry.stage) ||
            !reader.string(entry.profile))
            return shader_package_result::failure(corrupt_package("truncated entry-point table"));
        for (auto& size : entry.thread_group_size)
            if (!reader.value(size)) return shader_package_result::failure(corrupt_package("truncated thread-group size"));
    }
    if (!read_count(count)) return shader_package_result::failure(corrupt_package("invalid parameter count"));
    reflection.parameters.resize(count);
    for (auto& parameter : reflection.parameters)
    {
        std::vector<std::uint8_t> default_bytes;
        if (!reader.value(parameter.id.value) || !reader.string(parameter.name) || !reader.value(parameter.type) ||
            !reader.value(parameter.offset) || !reader.value(parameter.size) || !reader.raw(default_bytes))
            return shader_package_result::failure(corrupt_package("truncated parameter table"));
        parameter.default_value.resize(default_bytes.size());
        std::transform(default_bytes.begin(), default_bytes.end(), parameter.default_value.begin(),
                       [](std::uint8_t value) { return static_cast<std::byte>(value); });
    }
    if (!read_count(count)) return shader_package_result::failure(corrupt_package("invalid resource count"));
    reflection.resources.resize(count);
    for (auto& resource : reflection.resources)
    {
        std::uint8_t writable{};
        if (!reader.string(resource.name) || !reader.value(resource.kind) || !reader.value(resource.set) ||
            !reader.value(resource.binding) || !reader.value(resource.count) || !reader.value(writable))
            return shader_package_result::failure(corrupt_package("truncated resource table"));
        resource.writable = writable != 0;
    }
    if (!read_count(count)) return shader_package_result::failure(corrupt_package("invalid pass count"));
    reflection.passes.resize(count);
    for (auto& pass : reflection.passes)
    {
        std::uint8_t generated{};
        if (!reader.value(pass.pass) || !reader.value(pass.entry_point.value) || !reader.value(generated))
            return shader_package_result::failure(corrupt_package("truncated pass table"));
        pass.generated = generated != 0;
    }
    if (!read_count(count)) return shader_package_result::failure(corrupt_package("invalid dependency count"));
    package.compiled.dependencies.resize(count);
    for (auto& dependency : package.compiled.dependencies)
        if (!reader.string(dependency.path) || !reader.hash(dependency.content_hash))
            return shader_package_result::failure(corrupt_package("truncated dependency table"));

    const auto read_location = [&](shader_source_location& location)
    { return reader.string(location.path) && reader.value(location.line) && reader.value(location.column) &&
             reader.string(location.graph_node_id); };
    if (!read_count(count)) return shader_package_result::failure(corrupt_package("invalid source-map count"));
    package.compiled.source_map.resize(count);
    for (auto& mapping : package.compiled.source_map)
        if (!reader.value(mapping.generated_line) || !read_location(mapping.source))
            return shader_package_result::failure(corrupt_package("truncated source-map table"));
    if (!read_count(count)) return shader_package_result::failure(corrupt_package("invalid diagnostic count"));
    package.compiled.diagnostics.resize(count);
    for (auto& diagnostic : package.compiled.diagnostics)
    {
        std::uint32_t include_count{};
        std::uint8_t has_permutation{};
        if (!reader.value(diagnostic.severity) || !reader.string(diagnostic.code) ||
            !reader.string(diagnostic.message) || !read_location(diagnostic.location) ||
            !read_count(include_count))
            return shader_package_result::failure(corrupt_package("truncated diagnostic table"));
        diagnostic.include_stack.resize(include_count);
        for (auto& location : diagnostic.include_stack)
            if (!read_location(location))
                return shader_package_result::failure(corrupt_package("truncated diagnostic include stack"));
        if (!reader.value(has_permutation))
            return shader_package_result::failure(corrupt_package("truncated diagnostic permutation"));
        if (has_permutation != 0)
        {
            shader_permutation_id permutation;
            if (!reader.value(permutation.value))
                return shader_package_result::failure(corrupt_package("truncated diagnostic permutation"));
            diagnostic.permutation = permutation;
        }
    }

    if (reader.remaining() != 0 || !package.id.valid() || !package.generation.valid() ||
        !package.permutation.valid() || package.compiled.bytecode.empty() || package.compiled.build_hash.empty())
        return shader_package_result::failure(corrupt_package("shader package validation failed"));
    return shader_package_result::success(std::move(package));
}

shader_compile_result shader_library_cache::compile_or_get(shader_compiler& compiler,
                                                           const shader_compile_request& request)
{
    auto dependencies = collect_dependencies(request);
    if (!dependencies) return shader_compile_result::failure(dependencies.error());
    const auto hash = request_hash(request, dependencies.value(), compiler.fingerprint());
    const auto key = shader_cache_key(request);
    if (const auto found = cache_.find(key); found != cache_.end() && found->second.request_hash == hash)
        return found->second.result;

    auto result = compiler.compile(request);
    if (result)
    {
        result.value().build_hash = hash;
        result.value().dependencies = std::move(dependencies.value());
        result.value().compiler_fingerprint = std::string(compiler.fingerprint());
        cache_.insert_or_assign(key, cached_shader{.result = result, .request_hash = hash});
    }
    return result;
}

bool shader_library_cache::source_changed(const shader_compile_request& request) const
{
    const auto found = cache_.find(shader_cache_key(request));
    if (found == cache_.end()) return true;
    const auto dependencies = collect_dependencies(request);
    if (!dependencies) return true;
    // Compiler fingerprint is already represented in the cached hash. Source
    // checks use the stored fingerprint so callers can cheaply poll hot reload.
    const auto fingerprint = found->second.result ? found->second.result.value().compiler_fingerprint : std::string{};
    return found->second.request_hash != request_hash(request, dependencies.value(), fingerprint);
}

void shader_library_cache::clear() noexcept
{
    cache_.clear();
}

std::size_t shader_library_cache::size() const noexcept
{
    return cache_.size();
}

std::string shader_package_library::key(shader_package_id id, shader_permutation_id permutation)
{
    return core::to_string(id) + ":" + std::to_string(permutation.representation());
}

shader_publication_status shader_package_library::publish(shader_package package, std::uint64_t retire_after_frame)
{
    if (!package.id.valid() || !package.permutation.valid() || !package.generation.valid() ||
        package.compiled.bytecode.empty() || package.compiled.build_hash.empty())
        return shader_publication_status::rejected_incompatible_layout;

    const auto package_key = key(package.id, package.permutation);
    std::unique_lock lock{mutex_};
    auto existing = active_.find(package_key);
    if (existing != active_.end())
    {
        if (existing->second.active.generation.representation() > package.generation.representation())
            return shader_publication_status::rejected_stale_generation;
        if (existing->second.active.generation == package.generation)
            return existing->second.active.compiled.build_hash == package.compiled.build_hash
                       ? shader_publication_status::unchanged
                       : shader_publication_status::rejected_stale_generation;
        retired_.push_back({.package = std::move(existing->second.active), .retire_after_frame = retire_after_frame});
        existing->second.active = std::move(package);
        existing->second.last_error.reset();
    }
    else
    {
        active_.emplace(package_key, entry{.active = std::move(package)});
    }
    return shader_publication_status::published;
}

void shader_package_library::report_failure(shader_package_id id, shader_permutation_id permutation,
                                            shader_compile_error error)
{
    std::unique_lock lock{mutex_};
    if (auto found = active_.find(key(id, permutation)); found != active_.end())
        found->second.last_error = std::move(error);
}

std::optional<shader_package> shader_package_library::find(shader_package_id id,
                                                           shader_permutation_id permutation) const
{
    std::shared_lock lock{mutex_};
    const auto found = active_.find(key(id, permutation));
    return found == active_.end() ? std::nullopt : std::optional<shader_package>{found->second.active};
}

std::optional<shader_package_snapshot> shader_package_library::snapshot(shader_package_id id,
                                                                         shader_permutation_id permutation) const
{
    std::shared_lock lock{mutex_};
    const auto found = active_.find(key(id, permutation));
    if (found == active_.end()) return std::nullopt;
    return shader_package_snapshot{.id = id,
                                   .permutation = permutation,
                                   .generation = found->second.active.generation,
                                   .build_hash = found->second.active.compiled.build_hash,
                                   .last_error = found->second.last_error};
}

void shader_package_library::collect(std::uint64_t completed_frame)
{
    std::unique_lock lock{mutex_};
    std::erase_if(retired_, [completed_frame](const retired_entry& entry)
                  { return entry.retire_after_frame <= completed_frame; });
}

void shader_package_library::clear()
{
    std::unique_lock lock{mutex_};
    active_.clear();
    retired_.clear();
}

std::size_t shader_package_library::active_count() const
{
    std::shared_lock lock{mutex_};
    return active_.size();
}

std::size_t shader_package_library::retired_count() const
{
    std::shared_lock lock{mutex_};
    return retired_.size();
}

} // namespace arc::render
