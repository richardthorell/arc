#include <arc/render/shader.h>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string_view>
#include <utility>

namespace
{

class request_counting_compiler final : public arc::render::shader_compiler
{
public:
    arc::render::shader_compile_result compile(const arc::render::shader_compile_request&) override
    {
        ++count;
        arc::render::shader_compile_output output;
        output.bytecode = {static_cast<std::uint8_t>(count)};
        output.compiler_fingerprint = std::string(fingerprint());
        return arc::render::shader_compile_result::success(std::move(output));
    }

    std::string_view fingerprint() const noexcept override
    {
        return "arc.tests/request-counting";
    }

    int count{};
};

} // namespace

TEST_CASE("shader cache separates backend-neutral compile options")
{
    const auto path = std::filesystem::temp_directory_path() / "arc_shader_compile_options_test.slang";
    {
        std::ofstream file(path);
        file << "[shader(\"fragment\")] float4 main() : SV_Target { return 1; }";
    }

    request_counting_compiler compiler;
    arc::render::shader_library_cache cache;
    arc::render::shader_compile_request request{.source_path = path.string(),
                                                .entry_point = "main",
                                                .profile = "spirv_1_5",
                                                .domain = arc::render::shader_domain::surface,
                                                .stage = arc::render::shader_stage::fragment,
                                                .target = arc::render::shader_target::spirv,
                                                .optimization = arc::render::shader_optimization::development};

    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 1);

    request.stage = arc::render::shader_stage::vertex;
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 2);

    request.optimization = arc::render::shader_optimization::performance;
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 3);

    request.static_switches.push_back(
        {.id = arc::render::make_shader_parameter_id("USE_FOG"), .name = "USE_FOG", .value = true});
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 4);

    request.library_version = "arc-shader-library/2";
    REQUIRE(cache.compile_or_get(compiler, request));
    REQUIRE(compiler.count == 5);

    std::filesystem::remove(path);
}
