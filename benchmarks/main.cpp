#include <arc/assets/assets.h>
#include <arc/assets/cook.h>
#include <arc/ecs/ecs.h>
#include <arc/framework/framework.h>
#include <arc/io/io.h>
#include <arc/jobs/jobs.h>
#include <arc/memory/memory.h>
#include <arc/persistence/persistence.h>
#include <arc/render/render.h>
#include <arc/scene/scene.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace arc::benchmarks
{
struct position
{
    float x{};
    float y{};
    float z{};
};
} // namespace arc::benchmarks

namespace arc::ecs
{
template <> struct component_traits<benchmarks::position>
{
    static constexpr bool reflected = true;
    static constexpr std::string_view canonical_name = "arc.benchmarks.position";
    static constexpr component_type_id id{0xa8f4f09681404c8bull, 0xa972e79e120df6b9ull};
    static constexpr std::array<component_field_descriptor, 3> fields{{
        {1, "x", "X", reflected_field_kind::floating_point},
        {2, "y", "Y", reflected_field_kind::floating_point},
        {3, "z", "Z", reflected_field_kind::floating_point},
    }};
    static constexpr component_descriptor descriptor{id,
                                                     canonical_name,
                                                     "Benchmark position",
                                                     1,
                                                     sizeof(benchmarks::position),
                                                     alignof(benchmarks::position),
                                                     fields,
                                                     false,
                                                     false};
};
} // namespace arc::ecs

namespace
{
using clock_type = std::chrono::steady_clock;

struct options
{
    std::filesystem::path baseline;
    std::filesystem::path output{"benchmark-results.json"};
    double minimum_sample_ms{250.0};
    std::size_t warmups{2};
    std::size_t samples{7};
    double threshold{0.20};
};

struct measurement
{
    std::string name;
    double median_nanoseconds{};
    double normalized{};
    double baseline{};
    bool regressed{};
    std::vector<double> samples;
};

std::uint64_t observable_sink{};

bool parse_number(std::string_view text, double& value)
{
    try
    {
        std::size_t consumed{};
        value = std::stod(std::string(text), &consumed);
        return consumed == text.size() && std::isfinite(value);
    }
    catch (...)
    {
        return false;
    }
}

bool parse_size(std::string_view text, std::size_t& value)
{
    double number{};
    if (!parse_number(text, number) || number < 0.0) return false;
    value = static_cast<std::size_t>(number);
    return true;
}

bool parse_options(int argc, char** argv, options& result)
{
    for (int index = 1; index < argc; ++index)
    {
        const std::string_view argument(argv[index]);
        const auto next = [&]() -> std::string_view
        { return index + 1 < argc ? std::string_view(argv[++index]) : std::string_view{}; };
        if (argument == "--baseline")
            result.baseline = next();
        else if (argument == "--output")
            result.output = next();
        else if (argument == "--minimum-sample-ms")
        {
            if (!parse_number(next(), result.minimum_sample_ms)) return false;
        }
        else if (argument == "--warmups")
        {
            if (!parse_size(next(), result.warmups)) return false;
        }
        else if (argument == "--samples")
        {
            if (!parse_size(next(), result.samples)) return false;
        }
        else if (argument == "--threshold")
        {
            if (!parse_number(next(), result.threshold)) return false;
        }
        else
            return false;
    }
    return !result.baseline.empty() && result.samples > 0;
}

std::map<std::string, double> read_baselines(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) return {};
    std::map<std::string, double> result;
    std::string line;
    while (std::getline(input, line))
    {
        const auto key_begin = line.find('"');
        const auto key_end = key_begin == std::string::npos ? key_begin : line.find('"', key_begin + 1);
        const auto colon = key_end == std::string::npos ? key_end : line.find(':', key_end + 1);
        if (key_begin == std::string::npos || key_end == std::string::npos || colon == std::string::npos) continue;
        const auto value_begin = line.find_first_of("-0123456789", colon + 1);
        if (value_begin == std::string::npos) continue;
        const auto value_end = line.find_first_not_of("-.0123456789eE+", value_begin);
        double value{};
        if (parse_number(std::string_view(line).substr(value_begin, value_end - value_begin), value))
            result.emplace(line.substr(key_begin + 1, key_end - key_begin - 1), value);
    }
    return result;
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

double run_sample(const std::function<std::uint64_t()>& workload, double minimum_ms)
{
    std::size_t iterations{};
    std::uint64_t sink{};
    const auto begin = clock_type::now();
    double elapsed{};
    do
    {
        sink ^= workload() + iterations;
        ++iterations;
        elapsed = std::chrono::duration<double, std::nano>(clock_type::now() - begin).count();
    } while (elapsed < minimum_ms * 1'000'000.0);
    observable_sink ^= sink;
    return elapsed / static_cast<double>(iterations);
}

arc::persistence::archive_document make_document()
{
    arc::persistence::archive_document document;
    document.id = {1, 2};
    document.name = "ARC benchmark";
    for (std::uint64_t index = 0; index < 64; ++index)
    {
        arc::persistence::archive_entity_record entity;
        entity.id = {index + 10, index + 100};
        arc::persistence::archive_component_record component;
        component.type = arc::ecs::component_type_id{0x1111, 0x2222};
        component.name = "Benchmark";
        arc::persistence::archive_value value;
        value.kind = arc::persistence::archive_value_kind::floating_point;
        value.floating_point = static_cast<double>(index);
        component.fields.push_back({1, "value", std::move(value), true});
        entity.components.push_back(std::move(component));
        document.entities.push_back(std::move(entity));
    }
    return document;
}

arc::render::render_graph make_graph()
{
    arc::render::render_graph graph;
    auto previous = graph.add_resource({
        .name = "resource-0",
        .kind = arc::render::render_resource_kind::color_texture,
        .extent = {1920, 1080, 1},
        .format = arc::render::render_format::rgba16_float,
        .imported = true,
    });
    for (std::uint32_t index = 0; index < 12; ++index)
    {
        const auto next = graph.add_resource({
            .name = "resource-" + std::to_string(index + 1),
            .kind = arc::render::render_resource_kind::color_texture,
            .extent = {1920, 1080, 1},
            .format = arc::render::render_format::rgba16_float,
        });
        graph.add_pass({
            .name = "pass-" + std::to_string(index),
            .reads = {{.handle = previous,
                       .kind = arc::render::render_resource_kind::color_texture,
                       .usage = arc::render::render_resource_usage::sampled}},
            .writes = {{.handle = next,
                        .kind = arc::render::render_resource_kind::color_texture,
                        .usage = arc::render::render_resource_usage::color_attachment,
                        .write = true}},
        });
        previous = next;
    }
    return graph;
}
} // namespace

int main(int argc, char** argv)
{
    options config;
    if (!parse_options(argc, argv, config))
    {
        std::cerr << "Usage: arc-benchmarks --baseline FILE [--output FILE] "
                     "[--minimum-sample-ms MS] [--warmups N] [--samples N] [--threshold FRACTION]\n";
        return 2;
    }
    const auto baselines = read_baselines(config.baseline);
    if (baselines.empty())
    {
        std::cerr << "No benchmark baselines found in " << config.baseline << '\n';
        return 2;
    }
    std::cerr << "ARC benchmark setup: ECS\n";

    arc::ecs::world ecs_world;
    ecs_world.prepare_typed_query<arc::ecs::query_read<arc::benchmarks::position>>();
    for (std::uint32_t index = 0; index < 4096; ++index)
        ecs_world.emplace<arc::benchmarks::position>(ecs_world.create(),
                                                     arc::benchmarks::position{static_cast<float>(index), 1.0f, 2.0f});

    arc::jobs::job_system jobs({
        .worker_count = 4,
        .run_inline = false,
        .io_worker_count = 0,
        .enable_render_thread = false,
    });
    std::cerr << "ARC benchmark setup: assets\n";

    std::array<std::byte, 64 * 1024> hash_input{};
    for (std::size_t index = 0; index < hash_input.size(); ++index)
        hash_input[index] = static_cast<std::byte>(index & 0xffu);

    arc::memory::memory_system memory;
    arc::jobs::job_system asset_jobs(arc::jobs::job_system::single_threaded_config());
    arc::io::async_file_service files(asset_jobs);
    const auto temporary_root = std::filesystem::temp_directory_path() /
                                ("arc-benchmark-assets-" + arc::assets::to_string(arc::assets::generate_asset_guid()));
    std::filesystem::create_directories(temporary_root / "assets");
    arc::assets::asset_manager assets(
        {
            .project_root = temporary_root,
            .asset_root = temporary_root / "assets",
            .cache_root = temporary_root / ".arc/cache",
            .enable_source_monitor = false,
        },
        asset_jobs, files, memory);
    arc::framework::runtime_service_registry services;
    arc::framework::runtime_service_context service_context(services);
    assets.on_start(service_context);
    std::vector<arc::assets::asset_guid> asset_guids;
    for (std::uint64_t index = 0; index < 1024; ++index)
    {
        const arc::assets::asset_guid guid{0xabcdef00u + index, 0x12345678u + index};
        asset_guids.push_back(guid);
        assets.register_virtual_asset(guid, arc::assets::asset_types::material,
                                      arc::assets::asset_payload::make(arc::assets::asset_types::material,
                                                                       std::make_shared<const std::uint64_t>(index)),
                                      "benchmark-" + std::to_string(index));
    }
    arc::assets::derived_data_cache cache({.root = temporary_root / ".arc/cache"});
    arc::assets::cache_error cache_error;
    const auto cached_hash = arc::assets::hash_bytes(hash_input);
    if (!cache.put_blob(cached_hash, hash_input, cache_error))
    {
        std::cerr << "Failed to prepare local CAS benchmark: " << cache_error.message << '\n';
        assets.on_shutdown(service_context);
        return 2;
    }
    std::cerr << "ARC benchmark setup: workloads\n";

    const auto document = make_document();
    arc::persistence::component_persistence_registry persistence_components;
    persistence_components.freeze();
    arc::ecs::world render_scene_world;
    arc::render::renderer scene_renderer;
    const auto camera = render_scene_world.create();
    arc::scene::transform_component camera_transform;
    camera_transform.position = {0.0f, 30.0f, 40.0f};
    render_scene_world.emplace<arc::scene::transform_component>(camera, camera_transform);
    render_scene_world.emplace<arc::scene::camera_component>(camera);
    for (std::uint32_t index = 0; index < 2048; ++index)
    {
        const auto value = render_scene_world.create();
        arc::scene::transform_component transform;
        transform.position = {static_cast<float>(index % 64) - 32.0f, 0.0f, -static_cast<float>(index / 64)};
        render_scene_world.emplace<arc::scene::transform_component>(value, transform);
        render_scene_world.emplace<arc::scene::mesh_renderer_component>(
            value, arc::scene::mesh_renderer_component{.mesh = {.index = index % 16 + 1, .generation = 1}});
    }
    arc::scene::prepare_render_scene_queries(render_scene_world);
    std::uint64_t render_frame{};
    const std::vector<std::pair<std::string, std::function<std::uint64_t()>>> workloads{
        {"ecs.prepared-query",
         [&]
         {
             std::uint64_t sum{};
             for (const auto entity : ecs_world.query<arc::ecs::query_read<arc::benchmarks::position>>())
                 sum += entity.index;
             return sum;
         }},
        {"ecs.command-flush",
         [&]
         {
             arc::ecs::world command_world;
             arc::ecs::entity_command_buffer commands;
             for (std::size_t index = 0; index < 64; ++index)
             {
                 const auto entity = commands.create();
                 commands.add<arc::benchmarks::position>(entity, {});
                 commands.destroy(entity);
             }
             std::vector<arc::ecs::entity_command_buffer*> buffers{&commands};
             const auto result = arc::ecs::entity_command_buffer::flush_ordered(command_world, buffers);
             return static_cast<std::uint64_t>(result.applied);
         }},
        {"jobs.dispatch",
         [&]
         {
             std::array<arc::jobs::job_handle, 64> handles;
             for (auto& handle : handles)
                 handle = jobs.submit([] {});
             for (const auto& handle : handles)
                 handle.wait();
             return std::uint64_t{64};
         }},
        {"jobs.dependencies",
         [&]
         {
             std::array<arc::jobs::job_handle, 16> leaves;
             for (auto& leaf : leaves)
                 leaf = jobs.submit([] {});
             const auto joined = jobs.submit(
                 {
                     .name = "benchmark fan-in",
                     .dependency_view = leaves,
                 },
                 [] {});
             joined.wait();
             return std::uint64_t{17};
         }},
        {"assets.sha256",
         [&]
         {
             const auto hash = arc::assets::hash_bytes(hash_input);
             std::uint64_t value{};
             for (std::size_t index = 0; index < sizeof(value); ++index)
                 value |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(hash.bytes[index])) << (index * 8u);
             return value;
         }},
        {"assets.registry-lookup",
         [&]
         {
             std::uint64_t found{};
             for (const auto guid : asset_guids)
                 found += assets.find(guid).has_value() ? 1u : 0u;
             return found;
         }},
        {"assets.cas-hit",
         [&]
         {
             arc::assets::cache_error error;
             const auto blob = cache.get_blob(cached_hash, error);
             return blob ? blob->bytes.size() : 0u;
         }},
        {"persistence.json",
         [&]
         {
             const auto encoded = arc::persistence::write_reflected_json(document, false);
             if (!encoded) return std::size_t{};
             const auto decoded = arc::persistence::read_reflected_json(encoded.value(), persistence_components);
             return decoded.succeeded() ? encoded.value().size() + decoded.document.entities.size() : 0u;
         }},
        {"persistence.binary",
         [&]
         {
             const auto encoded = arc::persistence::write_tagged_binary(document, "benchmark");
             if (!encoded) return std::size_t{};
             const auto decoded = arc::persistence::read_tagged_binary(encoded.value(), persistence_components);
             return decoded.succeeded() ? encoded.value().size() + decoded.document.entities.size() : 0u;
         }},
        {"render.scene-extraction",
         [&]
         {
             const auto extracted = arc::scene::render_scene(render_scene_world, scene_renderer, 1920, 1080);
             const auto frame = scene_renderer.frame_queue().commit(++render_frame);
             return extracted.submitted_draw_count + frame.events.size();
         }},
        {"render.graph-compile",
         [&]
         {
             const auto graph = make_graph();
             return graph.compile().transitions.size();
         }},
    };

    const auto calibration = []
    {
        std::uint64_t value{0x9e3779b97f4a7c15ull};
        for (std::uint64_t index = 0; index < 4096; ++index)
            value = (value ^ (index + 0x9e3779b97f4a7c15ull)) * 0xbf58476d1ce4e5b9ull;
        return value;
    };
    const double calibration_ns = run_sample(calibration, config.minimum_sample_ms);
    std::cerr << "ARC benchmark calibration complete\n";

    std::vector<measurement> results;
    bool failed{};
    for (const auto& [name, workload] : workloads)
    {
        std::cerr << "Running " << name << '\n';
        for (std::size_t index = 0; index < config.warmups; ++index)
            static_cast<void>(run_sample(workload, std::min(25.0, config.minimum_sample_ms)));
        measurement value;
        value.name = name;
        for (std::size_t index = 0; index < config.samples; ++index)
            value.samples.push_back(run_sample(workload, config.minimum_sample_ms));
        value.median_nanoseconds = median(value.samples);
        value.normalized = value.median_nanoseconds / calibration_ns;
        const auto baseline = baselines.find(name);
        if (baseline == baselines.end())
        {
            std::cerr << "Missing baseline for " << name << '\n';
            failed = true;
        }
        else
        {
            value.baseline = baseline->second;
            if (value.normalized > value.baseline * (1.0 + config.threshold))
            {
                std::vector<double> confirmation_samples;
                confirmation_samples.reserve(config.samples);
                for (std::size_t index = 0; index < config.samples; ++index)
                    confirmation_samples.push_back(run_sample(workload, config.minimum_sample_ms));
                const double confirmation = median(std::move(confirmation_samples)) / calibration_ns;
                value.regressed = confirmation > value.baseline * (1.0 + config.threshold);
                failed = failed || value.regressed;
            }
        }
        std::cout << name << ": " << value.normalized << " normalized (baseline " << value.baseline << ")\n";
        results.push_back(std::move(value));
    }

    assets.on_shutdown(service_context);
    std::error_code cleanup_error;
    std::filesystem::remove_all(temporary_root, cleanup_error);

    std::ofstream output(config.output, std::ios::binary | std::ios::trunc);
    output << "{\n  \"format\": \"arc-benchmark-results\",\n  \"version\": 1,\n"
              "  \"calibrationNanoseconds\": "
           << calibration_ns << ",\n  \"observer\": " << observable_sink << ",\n  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index)
    {
        const auto& value = results[index];
        output << "    {\"name\":\"" << value.name << "\",\"medianNanoseconds\":" << value.median_nanoseconds
               << ",\"normalized\":" << value.normalized << ",\"baseline\":" << value.baseline
               << ",\"regressed\":" << (value.regressed ? "true" : "false") << ",\"sampleNanoseconds\":[";
        for (std::size_t sample = 0; sample < value.samples.size(); ++sample)
        {
            output << value.samples[sample];
            if (sample + 1 != value.samples.size()) output << ',';
        }
        output << "]}";
        output << (index + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    return failed ? 1 : 0;
}
