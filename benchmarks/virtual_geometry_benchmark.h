#pragma once

#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace arc::benchmarks
{

enum class virtual_geometry_corpus_scale : std::uint8_t
{
    disabled,
    ci,
    developer,
    massive
};

struct virtual_geometry_view_capture
{
    std::string name;
    std::uint64_t visible_cluster_fingerprint{};
    std::uint64_t visible_triangles{};
    std::uint32_t visible_clusters{};
    std::uint32_t requested_pages{};
    std::uint32_t parent_fallbacks{};
    std::uint32_t traversal_overflow{};
};

struct virtual_geometry_corpus_result
{
    std::string name;
    std::uint32_t source_vertices{};
    std::uint64_t source_triangles{};
    std::uint64_t source_bytes{};
    std::uint32_t cluster_count{};
    std::uint32_t hierarchy_nodes{};
    std::uint32_t hierarchy_levels{};
    std::uint32_t page_count{};
    std::uint32_t root_page_count{};
    std::uint64_t decoded_page_bytes{};
    std::uint64_t stored_page_bytes{};
    std::uint64_t artifact_bytes{};
    std::uint64_t hierarchy_fingerprint{};
    std::uint64_t process_peak_resident_bytes{};
    double cook_milliseconds{};
    double artifact_encode_milliseconds{};
    double reference_traversal_milliseconds{};
    bool gpu_timings_available{};
    double gpu_traversal_milliseconds{};
    double gpu_raster_milliseconds{};
    double gpu_material_milliseconds{};
    std::vector<virtual_geometry_view_capture> captures;
};

[[nodiscard]] bool parse_virtual_geometry_corpus_scale(std::string_view text,
                                                       virtual_geometry_corpus_scale& output) noexcept;
[[nodiscard]] std::string_view to_string(virtual_geometry_corpus_scale scale) noexcept;

/** Build and inspect the deterministic generated corpus for one benchmark scale. */
[[nodiscard]] virtual_geometry_corpus_result run_virtual_geometry_corpus(virtual_geometry_corpus_scale scale);

/** Append the stable JSON member consumed by benchmark trend tooling. */
void write_virtual_geometry_corpus_json(std::ostream& output, const virtual_geometry_corpus_result& result,
                                        std::string_view indent = "  ");

} // namespace arc::benchmarks
