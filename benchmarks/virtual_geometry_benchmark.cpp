#include "virtual_geometry_benchmark.h"

#include <arc/render/render.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <ostream>
#include <span>
#include <vector>

#if defined(_WIN32)
#include <Windows.h>
#include <Psapi.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace arc::benchmarks
{
namespace
{
using clock_type = std::chrono::steady_clock;

struct corpus_definition
{
    std::string_view name;
    std::uint32_t side{};
};

corpus_definition definition(virtual_geometry_corpus_scale scale) noexcept
{
    switch (scale)
    {
        case virtual_geometry_corpus_scale::ci:
            return {"generated-wave-grid-ci", 65u};
        case virtual_geometry_corpus_scale::developer:
            return {"generated-wave-grid-developer", 257u};
        case virtual_geometry_corpus_scale::massive:
            return {"generated-wave-grid-10m", 2305u};
        case virtual_geometry_corpus_scale::disabled:
            break;
    }
    return {"disabled", 0u};
}

arc::render::mesh_data make_wave_grid(const corpus_definition& corpus)
{
    arc::render::mesh_data mesh;
    mesh.name = std::string(corpus.name);
    mesh.material_index = 1u;
    const auto side = corpus.side;
    mesh.vertices.resize(static_cast<std::size_t>(side) * side);
    for (std::uint32_t z = 0; z < side; ++z)
        for (std::uint32_t x = 0; x < side; ++x)
        {
            auto& vertex = mesh.vertices[static_cast<std::size_t>(z) * side + x];
            const auto normalized_x = static_cast<float>(x) / static_cast<float>(side - 1u);
            const auto normalized_z = static_cast<float>(z) / static_cast<float>(side - 1u);
            vertex.position[0] = (normalized_x - 0.5f) * 512.0f;
            vertex.position[1] = std::sin(normalized_x * 31.0f) * std::cos(normalized_z * 27.0f) * 12.0f +
                                 std::sin((normalized_x + normalized_z) * 113.0f) * 0.75f;
            vertex.position[2] = (normalized_z - 0.5f) * 512.0f;
            vertex.normal[1] = 1.0f;
            vertex.tangent[0] = 1.0f;
            vertex.tangent[3] = 1.0f;
            vertex.texcoord[0] = normalized_x;
            vertex.texcoord[1] = normalized_z;
            std::fill(std::begin(vertex.color), std::end(vertex.color), 1.0f);
        }
    const auto quad_count = static_cast<std::uint64_t>(side - 1u) * (side - 1u);
    mesh.indices.reserve(static_cast<std::size_t>(quad_count * 6u));
    for (std::uint32_t z = 0; z + 1u < side; ++z)
        for (std::uint32_t x = 0; x + 1u < side; ++x)
        {
            const auto i0 = z * side + x;
            const auto i1 = i0 + 1u;
            const auto i2 = i0 + side;
            const auto i3 = i2 + 1u;
            mesh.indices.insert(mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
        }
    return mesh;
}

template <class T> void hash_value(std::uint64_t& hash, const T& value) noexcept
{
    const auto bytes = std::as_bytes(std::span(&value, 1));
    for (const auto byte : bytes)
    {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ull;
    }
}

std::uint64_t hierarchy_fingerprint(const arc::render::virtual_mesh_data& geometry) noexcept
{
    std::uint64_t hash{1469598103934665603ull};
    for (const auto& cluster : geometry.clusters)
    {
        hash_value(hash, cluster.first_index);
        hash_value(hash, cluster.index_count);
        hash_value(hash, cluster.geometric_error);
        hash_value(hash, cluster.hierarchy_node);
        hash_value(hash, cluster.page_index);
    }
    for (const auto& node : geometry.lod_nodes)
    {
        hash_value(hash, node.first_cluster);
        hash_value(hash, node.cluster_count);
        hash_value(hash, node.first_child);
        hash_value(hash, node.child_count);
        hash_value(hash, node.parent);
        hash_value(hash, node.error);
    }
    for (const auto child : geometry.hierarchy_children)
        hash_value(hash, child);
    for (const auto root : geometry.root_nodes)
        hash_value(hash, root);
    for (const auto byte : geometry.page_payload)
    {
        hash ^= std::to_integer<std::uint8_t>(byte);
        hash *= 1099511628211ull;
    }
    return hash;
}

std::uint64_t visible_fingerprint(std::span<const std::uint32_t> visible) noexcept
{
    std::uint64_t hash{1469598103934665603ull};
    for (const auto value : visible)
        hash_value(hash, value);
    return hash;
}

std::uint64_t peak_resident_bytes() noexcept
{
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters)))
        return counters.PeakWorkingSetSize;
#elif defined(__unix__) || defined(__APPLE__)
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
#if defined(__APPLE__)
        return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
        return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024u;
#endif
    }
#endif
    return 0u;
}

virtual_geometry_view_capture capture_view(std::string name, const arc::render::virtual_mesh_data& geometry,
                                           std::span<const std::uint8_t> resident,
                                           arc::render::virtual_geometry_reference_view view)
{
    const auto traversal = arc::render::traverse_virtual_geometry_reference(geometry, resident, view);
    std::uint64_t triangles{};
    for (const auto cluster : traversal.visible_clusters)
        if (cluster < geometry.clusters.size()) triangles += geometry.clusters[cluster].triangle_count;
    return {.name = std::move(name),
            .visible_cluster_fingerprint = visible_fingerprint(traversal.visible_clusters),
            .visible_triangles = triangles,
            .visible_clusters = static_cast<std::uint32_t>(traversal.visible_clusters.size()),
            .requested_pages = static_cast<std::uint32_t>(traversal.requested_pages.size()),
            .parent_fallbacks = traversal.parent_fallbacks};
}

bool sequence_occluded(const arc::math::vector3f&, float, void* user_data)
{
    return *static_cast<const bool*>(user_data);
}

bool sequence_refined(std::uint32_t node_index, void* user_data)
{
    const auto& history = *static_cast<const std::vector<std::uint8_t>*>(user_data);
    return node_index < history.size() && history[node_index] != 0u;
}

double reduction(std::uint64_t baseline, std::uint64_t measured) noexcept
{
    if (baseline == 0u) return 0.0;
    return 1.0 - static_cast<double>(measured) / static_cast<double>(baseline);
}

virtual_geometry_sequence_capture capture_occlusion_sequence(const arc::render::virtual_mesh_data& geometry,
                                                              std::span<const std::uint8_t> resident,
                                                              arc::render::virtual_geometry_reference_view view)
{
    virtual_geometry_sequence_capture capture{.name = "deterministic-occluder", .frames = 8u};
    for (std::uint32_t frame = 0u; frame < capture.frames; ++frame)
    {
        const auto baseline = arc::render::traverse_virtual_geometry_reference(geometry, resident, view);
        capture.baseline_traversed_nodes += baseline.traversed_nodes;
        capture.baseline_rasterized_clusters += baseline.visible_clusters.size();

        bool previous_occluded = frame < capture.frames - 2u;
        bool current_occluded = frame == capture.frames - 2u;
        auto previous_view = view;
        previous_view.traversal_phase = arc::render::virtual_geometry_traversal_phase::previous_hzb;
        previous_view.occluded = &sequence_occluded;
        previous_view.occlusion_user_data = &previous_occluded;
        const auto previous = arc::render::traverse_virtual_geometry_reference(geometry, resident, previous_view);
        capture.two_phase_traversed_nodes += previous.traversed_nodes;
        capture.two_phase_rasterized_clusters += previous.visible_clusters.size();
        capture.previous_hzb_rejections += previous.previous_hzb_rejected;

        if (!previous.visible_clusters.empty())
        {
            auto current_view = view;
            current_view.traversal_phase = arc::render::virtual_geometry_traversal_phase::current_hzb;
            current_view.occluded = &sequence_occluded;
            current_view.occlusion_user_data = &current_occluded;
            const auto current = arc::render::traverse_virtual_geometry_reference(geometry, resident, current_view);
            capture.two_phase_traversed_nodes += current.traversed_nodes;
            capture.two_phase_rasterized_clusters += current.visible_clusters.size();
            capture.current_hzb_rejections += current.current_hzb_rejected;
            const bool expected_visible = !current_occluded;
            capture.visible_geometry_preserved =
                capture.visible_geometry_preserved &&
                (expected_visible ? current.visible_clusters == baseline.visible_clusters
                                  : current.visible_clusters.empty());
        }
        else if (!previous_occluded)
            capture.visible_geometry_preserved = false;
    }
    capture.traversal_work_reduction =
        reduction(capture.baseline_traversed_nodes, capture.two_phase_traversed_nodes);
    capture.raster_work_reduction =
        reduction(capture.baseline_rasterized_clusters, capture.two_phase_rasterized_clusters);
    return capture;
}

virtual_geometry_sequence_capture capture_threshold_sequence(const arc::render::virtual_mesh_data& geometry,
                                                              std::span<const std::uint8_t> resident,
                                                              arc::render::virtual_geometry_reference_view view)
{
    virtual_geometry_sequence_capture capture{.name = "slow-threshold", .frames = 10u};
    if (geometry.root_nodes.empty()) return capture;
    const auto root_index = geometry.root_nodes.front();
    if (root_index >= geometry.lod_nodes.size()) return capture;
    const auto& root = geometry.lod_nodes[root_index];
    const auto distance = std::sqrt((std::max)(arc::math::length_squared(
                                                  arc::math::sub(view.camera_position, root.sphere_center)),
                                              1.0e-12f));
    const auto nearest_distance = (std::max)(distance - root.sphere_radius, 1.0e-4f);
    const auto unit_scale = root.error > 0.0f ? nearest_distance / root.error : view.projection_scale;
    constexpr std::array multipliers{0.95f, 1.05f, 0.98f, 1.02f, 1.09f,
                                     1.11f, 1.05f, 0.95f, 0.91f, 0.89f};
    std::vector<std::uint8_t> history(geometry.lod_nodes.size());
    bool refined{};
    for (const auto multiplier : multipliers)
    {
        view.projection_scale = unit_scale * multiplier;
        view.refined_last_frame = &sequence_refined;
        view.refinement_history_user_data = &history;
        const auto frame = arc::render::traverse_virtual_geometry_reference(geometry, resident, view);
        capture.two_phase_traversed_nodes += frame.traversed_nodes;
        capture.two_phase_rasterized_clusters += frame.visible_clusters.size();
        capture.hysteresis_decisions +=
            frame.hysteresis_refine_suppressed + frame.hysteresis_coarsen_suppressed;
        const bool now_refined = std::find(frame.refined_nodes.begin(), frame.refined_nodes.end(), root_index) !=
                                 frame.refined_nodes.end();
        if (now_refined != refined) ++capture.refinement_transitions;
        refined = now_refined;
        std::fill(history.begin(), history.end(), std::uint8_t{0});
        for (const auto node : frame.refined_nodes)
            if (node < history.size()) history[node] = 1u;
    }
    return capture;
}

void write_capture(std::ostream& output, const virtual_geometry_view_capture& capture, std::string_view indent)
{
    output << indent << "{\"name\":\"" << capture.name << "\",\"visibleClusterFingerprint\":\"0x" << std::hex
           << capture.visible_cluster_fingerprint << std::dec << "\",\"visibleTriangles\":" << capture.visible_triangles
           << ",\"visibleClusters\":" << capture.visible_clusters << ",\"requestedPages\":" << capture.requested_pages
           << ",\"parentFallbacks\":" << capture.parent_fallbacks
           << ",\"traversalOverflow\":" << capture.traversal_overflow << '}';
}

void write_sequence(std::ostream& output, const virtual_geometry_sequence_capture& capture, std::string_view indent)
{
    output << indent << "{\"name\":\"" << capture.name << "\",\"frames\":" << capture.frames
           << ",\"baselineTraversedNodes\":" << capture.baseline_traversed_nodes
           << ",\"twoPhaseTraversedNodes\":" << capture.two_phase_traversed_nodes
           << ",\"baselineRasterizedClusters\":" << capture.baseline_rasterized_clusters
           << ",\"twoPhaseRasterizedClusters\":" << capture.two_phase_rasterized_clusters
           << ",\"previousHzbRejections\":" << capture.previous_hzb_rejections
           << ",\"currentHzbRejections\":" << capture.current_hzb_rejections
           << ",\"hysteresisDecisions\":" << capture.hysteresis_decisions
           << ",\"refinementTransitions\":" << capture.refinement_transitions
           << ",\"traversalWorkReduction\":" << capture.traversal_work_reduction
           << ",\"rasterWorkReduction\":" << capture.raster_work_reduction
           << ",\"visibleGeometryPreserved\":" << (capture.visible_geometry_preserved ? "true" : "false") << '}';
}
} // namespace

bool parse_virtual_geometry_corpus_scale(std::string_view text, virtual_geometry_corpus_scale& output) noexcept
{
    if (text == "off" || text == "disabled")
        output = virtual_geometry_corpus_scale::disabled;
    else if (text == "ci")
        output = virtual_geometry_corpus_scale::ci;
    else if (text == "developer")
        output = virtual_geometry_corpus_scale::developer;
    else if (text == "massive")
        output = virtual_geometry_corpus_scale::massive;
    else
        return false;
    return true;
}

std::string_view to_string(virtual_geometry_corpus_scale scale) noexcept
{
    switch (scale)
    {
        case virtual_geometry_corpus_scale::disabled:
            return "disabled";
        case virtual_geometry_corpus_scale::ci:
            return "ci";
        case virtual_geometry_corpus_scale::developer:
            return "developer";
        case virtual_geometry_corpus_scale::massive:
            return "massive";
    }
    return "disabled";
}

virtual_geometry_corpus_result run_virtual_geometry_corpus(virtual_geometry_corpus_scale scale)
{
    const auto corpus = definition(scale);
    virtual_geometry_corpus_result result;
    result.name = std::string(corpus.name);
    if (scale == virtual_geometry_corpus_scale::disabled) return result;

    auto source = make_wave_grid(corpus);
    result.source_vertices = static_cast<std::uint32_t>(source.vertices.size());
    result.source_triangles = source.indices.size() / 3u;
    result.source_bytes =
        source.vertices.size() * sizeof(arc::render::mesh_vertex) + source.indices.size() * sizeof(std::uint32_t);

    const auto cook_begin = clock_type::now();
    const auto geometry = arc::render::build_virtual_mesh(source, {.build_conventional_lods = false});
    result.cook_milliseconds = std::chrono::duration<double, std::milli>(clock_type::now() - cook_begin).count();
    result.cluster_count = geometry.stats.cluster_count;
    result.hierarchy_nodes = static_cast<std::uint32_t>(geometry.lod_nodes.size());
    result.hierarchy_levels = geometry.stats.hierarchy_level_count;
    result.page_count = geometry.stats.page_count;
    result.root_page_count = geometry.stats.root_page_count;
    result.decoded_page_bytes = geometry.stats.uncompressed_page_bytes;
    result.stored_page_bytes = geometry.stats.compressed_page_bytes;
    result.hierarchy_fingerprint = hierarchy_fingerprint(geometry);

    const std::array artifact_sources{arc::render::virtual_geometry_artifact_source{
        .name = source.name, .material_index = source.material_index, .geometry = &geometry}};
    const auto encode_begin = clock_type::now();
    const auto artifact = arc::render::encode_virtual_geometry_artifact(
        artifact_sources, result.hierarchy_fingerprint,
        {.page_order = arc::render::virtual_geometry_artifact_page_order::spatial_morton});
    result.artifact_encode_milliseconds =
        std::chrono::duration<double, std::milli>(clock_type::now() - encode_begin).count();
    if (artifact) result.artifact_bytes = artifact.value().size();

    std::vector<std::uint8_t> all_resident(geometry.pages.size(), 1u);
    std::vector<std::uint8_t> root_resident(geometry.pages.size());
    for (std::size_t page = 0; page < geometry.pages.size(); ++page)
        root_resident[page] = geometry.pages[page].root ? 1u : 0u;

    arc::render::virtual_geometry_reference_view view;
    view.projection_scale = 1080.0f;
    view.minimum_projected_radius = 0.0f;
    view.geometric_error_threshold = 1.0f;
    view.double_sided = true;
    view.camera_position = {0.0f, 160.0f, 300.0f};
    const auto traversal_begin = clock_type::now();
    result.captures.push_back(capture_view("flythrough-near", geometry, all_resident, view));
    view.camera_position = {180.0f, 240.0f, 420.0f};
    result.captures.push_back(capture_view("flythrough-oblique", geometry, all_resident, view));
    view.camera_position = {0.0f, 1200.0f, 2200.0f};
    result.captures.push_back(capture_view("flythrough-far", geometry, all_resident, view));
    view.camera_position = {-220.0f, 90.0f, -170.0f};
    view.camera_cut = true;
    result.captures.push_back(capture_view("camera-cut", geometry, all_resident, view));
    view.camera_cut = false;
    result.captures.push_back(capture_view("root-only-pressure", geometry, root_resident, view));
    view.camera_cut = false;
    view.camera_position = {0.0f, 160.0f, 300.0f};
    result.sequences.push_back(capture_occlusion_sequence(geometry, all_resident, view));
    result.sequences.push_back(capture_threshold_sequence(geometry, all_resident, view));
    result.reference_traversal_milliseconds =
        std::chrono::duration<double, std::milli>(clock_type::now() - traversal_begin).count();
    result.process_peak_resident_bytes = peak_resident_bytes();
    return result;
}

void write_virtual_geometry_corpus_json(std::ostream& output, const virtual_geometry_corpus_result& result,
                                        std::string_view indent)
{
    output << indent << "{\n";
    const std::string nested(indent.size() + 2u, ' ');
    output << nested << "\"name\":\"" << result.name << "\",\n"
           << nested << "\"sourceVertices\":" << result.source_vertices << ",\n"
           << nested << "\"sourceTriangles\":" << result.source_triangles << ",\n"
           << nested << "\"sourceBytes\":" << result.source_bytes << ",\n"
           << nested << "\"clusterCount\":" << result.cluster_count << ",\n"
           << nested << "\"hierarchyNodes\":" << result.hierarchy_nodes << ",\n"
           << nested << "\"hierarchyLevels\":" << result.hierarchy_levels << ",\n"
           << nested << "\"pageCount\":" << result.page_count << ",\n"
           << nested << "\"rootPageCount\":" << result.root_page_count << ",\n"
           << nested << "\"decodedPageBytes\":" << result.decoded_page_bytes << ",\n"
           << nested << "\"storedPageBytes\":" << result.stored_page_bytes << ",\n"
           << nested << "\"artifactBytes\":" << result.artifact_bytes << ",\n"
           << nested << "\"hierarchyFingerprint\":\"0x" << std::hex << result.hierarchy_fingerprint << std::dec
           << "\",\n"
           << nested << "\"processPeakResidentBytes\":" << result.process_peak_resident_bytes << ",\n"
           << nested << "\"cookMilliseconds\":" << result.cook_milliseconds << ",\n"
           << nested << "\"artifactEncodeMilliseconds\":" << result.artifact_encode_milliseconds << ",\n"
           << nested << "\"referenceTraversalMilliseconds\":" << result.reference_traversal_milliseconds << ",\n"
           << nested << "\"gpuTimings\":{\"available\":" << (result.gpu_timings_available ? "true" : "false")
           << ",\"traversalMilliseconds\":" << result.gpu_traversal_milliseconds
           << ",\"rasterMilliseconds\":" << result.gpu_raster_milliseconds
           << ",\"materialMilliseconds\":" << result.gpu_material_milliseconds << "},\n"
           << nested << "\"captures\":[\n";
    for (std::size_t index = 0; index < result.captures.size(); ++index)
    {
        write_capture(output, result.captures[index], std::string(nested.size() + 2u, ' '));
        output << (index + 1u == result.captures.size() ? "\n" : ",\n");
    }
    output << nested << "],\n" << nested << "\"sequences\":[\n";
    for (std::size_t index = 0; index < result.sequences.size(); ++index)
    {
        write_sequence(output, result.sequences[index], std::string(nested.size() + 2u, ' '));
        output << (index + 1u == result.sequences.size() ? "\n" : ",\n");
    }
    output << nested << "]\n" << indent << '}';
}

} // namespace arc::benchmarks
