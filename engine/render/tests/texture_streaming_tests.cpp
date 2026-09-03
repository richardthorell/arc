#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace
{

arc::render::texture_data make_rgba8_texture(std::uint32_t size)
{
    arc::render::texture_data texture;
    texture.name = "streaming-test";
    texture.width = size;
    texture.height = size;
    texture.dimension = arc::render::texture_dimension::texture_2d;
    texture.format = arc::render::texture_format::rgba8_unorm;
    texture.color_space = arc::render::texture_color_space::linear;
    texture.semantic = arc::render::texture_semantic::metallic_roughness;
    texture.array_layers = 1;
    for (std::uint32_t width = size, height = size;;
         width = std::max(1u, width / 2u), height = std::max(1u, height / 2u))
    {
        const auto offset = texture.pixels.size();
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x)
            {
                texture.pixels.push_back(static_cast<std::byte>(x & 0xffu));
                texture.pixels.push_back(static_cast<std::byte>(y & 0xffu));
                texture.pixels.push_back(static_cast<std::byte>((x + y) & 0xffu));
                texture.pixels.push_back(std::byte{0xff});
            }
        texture.mips.push_back(
            {.width = width, .height = height, .offset = offset, .size = texture.pixels.size() - offset});
        if (width == 1 && height == 1) break;
    }
    texture.mip_levels = static_cast<std::uint32_t>(texture.mips.size());
    return texture;
}

arc::render::streamed_texture_descriptor make_streamed_descriptor()
{
    arc::render::streamed_texture_descriptor descriptor;
    descriptor.texture = {.name = "residency-test",
                          .width = 8,
                          .height = 8,
                          .depth = 1,
                          .dimension = arc::render::texture_dimension::texture_2d,
                          .mip_levels = 4,
                          .format = arc::render::texture_format::rgba8_unorm,
                          .color_space = arc::render::texture_color_space::linear,
                          .semantic = arc::render::texture_semantic::metallic_roughness};
    descriptor.mode = arc::render::texture_streaming_mode::streamed_mips;
    descriptor.source = 77;
    descriptor.content_generation = 3;
    descriptor.artifact = {
        .schema_version = arc::render::texture_artifact_schema_version,
        .mode = arc::render::texture_streaming_mode::streamed_mips,
        .format = arc::render::texture_format::rgba8_unorm,
        .color_space = arc::render::texture_color_space::linear,
        .semantic = arc::render::texture_semantic::metallic_roughness,
        .width = 8,
        .height = 8,
        .mip_count = 4,
        .tail_first_mip = 2,
        .tile_size = arc::render::virtual_texture_tile_size,
        .tile_border = arc::render::virtual_texture_tile_border,
        .artifact_size = 16384,
        .mips = {{.width = 8, .height = 8, .offset = 4096, .stored_size = 256, .decoded_size = 256, .content_hash = 1},
                 {.width = 4, .height = 4, .offset = 8192, .stored_size = 64, .decoded_size = 64, .content_hash = 2},
                 {.width = 2, .height = 2, .offset = 12288, .stored_size = 16, .decoded_size = 16, .content_hash = 3},
                 {.width = 1, .height = 1, .offset = 16384, .stored_size = 4, .decoded_size = 4, .content_hash = 4}}};
    return descriptor;
}

} // namespace

TEST_CASE("texture artifacts are deterministic range-readable and include virtual companions")
{
    using namespace arc::render;
    const auto texture = make_rgba8_texture(256);
    const auto first = encode_texture_artifact(texture, texture_streaming_mode::virtual_tiles);
    const auto second = encode_texture_artifact(texture, texture_streaming_mode::virtual_tiles);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    REQUIRE(first.value() == second.value());

    const auto inspected = inspect_texture_artifact(first.value());
    REQUIRE(inspected.has_value());
    const auto& index = inspected.value();
    CHECK(index.mip_count == texture.mips.size());
    CHECK(index.tail_first_mip == 1);
    CHECK(index.tiles.size() == 4);
    CHECK(index.tile_size == 128);
    CHECK(index.tile_border == 4);
    for (const auto& mip : index.mips)
        CHECK(mip.offset % texture_artifact_alignment == 0);
    for (const auto& tile : index.tiles)
        CHECK(tile.offset % texture_artifact_alignment == 0);

    const auto tile = read_texture_artifact_tile(first.value(), index, 0);
    REQUIRE(tile.has_value());
    REQUIRE(tile.value().size() == 136u * 136u * 4u);
    CHECK(std::to_integer<std::uint8_t>(tile.value()[0]) == 252u);
    CHECK(std::to_integer<std::uint8_t>(tile.value()[1]) == 252u);

    auto corrupt = first.value();
    corrupt[static_cast<std::size_t>(index.mips[0].offset)] ^= std::byte{1};
    const auto corrupt_mip = read_texture_artifact_mip(corrupt, index, 0);
    REQUIRE_FALSE(corrupt_mip.has_value());
    CHECK(corrupt_mip.error().code == texture_artifact_error_code::integrity_failure);

    auto corrupt_header = first.value();
    corrupt_header[28] ^= std::byte{1};
    const auto rejected_header = inspect_texture_artifact(corrupt_header);
    REQUIRE_FALSE(rejected_header.has_value());
    CHECK(rejected_header.error().code == texture_artifact_error_code::integrity_failure);

    auto incomplete = texture;
    incomplete.mips.pop_back();
    const auto rejected_chain = encode_texture_artifact(incomplete, texture_streaming_mode::streamed_mips);
    REQUIRE_FALSE(rejected_chain.has_value());
    CHECK(rejected_chain.error().code == texture_artifact_error_code::invalid_data);
}

TEST_CASE("texture artifact sources constrain loose and package-relative ranges")
{
    using namespace arc;
    jobs::job_system jobs({.worker_count = 1, .run_inline = false, .io_worker_count = 1});
    io::async_file_service files(jobs);
    render::filesystem_texture_artifact_source source(files);
    const auto path = std::filesystem::temp_directory_path() / "arc_texture_package_range.bin";
    std::filesystem::remove(path);
    {
        std::ofstream output(path, std::ios::binary);
        const std::array<char, 12> bytes{'p', 'r', 'e', 'f', 'A', 'R', 'C', 'T', 'E', 'X', 'x', 'x'};
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    source.register_package_range(44, path, 4, 6);
    const auto payload = source.read_range(44, 0, 6).get();
    REQUIRE(payload.succeeded());
    REQUIRE(payload.value().size() == 6);
    CHECK(payload.value()[0] == std::byte{'A'});
    CHECK(payload.value()[5] == std::byte{'X'});
    const auto outside = source.read_range(44, 5, 2).get();
    REQUIRE_FALSE(outside.succeeded());
    CHECK(outside.error().code == io::file_error_code::invalid_range);
    source.unregister(44);
    const auto removed = source.read_range(44, 0, 1).get();
    REQUIRE_FALSE(removed.succeeded());
    std::filesystem::remove(path);
}

TEST_CASE("texture residency deduplicates demand rejects stale work and evicts unprotected fine mips")
{
    using namespace arc::render;
    texture_residency_manager residency({.gpu_budget_bytes = 40,
                                         .cpu_cache_budget_bytes = 4096,
                                         .upload_budget_per_frame = 4096,
                                         .maximum_requests_per_frame = 32,
                                         .protected_frame_count = 0});
    const texture_handle handle{4, 2};
    const auto descriptor = make_streamed_descriptor();
    residency.register_resource(handle, descriptor);
    residency.begin_frame(1);
    auto loads = residency.take_load_requests();
    REQUIRE(loads.size() == 2);
    CHECK(loads[0].mip == 3);
    CHECK(loads[1].mip == 2);
    for (const auto& load : loads)
    {
        residency.mark_loading(load);
        auto bytes = std::make_shared<const std::vector<std::byte>>(load.byte_size);
        residency.mark_uploading({.resource = handle,
                                  .content_generation = descriptor.content_generation,
                                  .kind = texture_subresource_kind::mip,
                                  .mip = load.mip,
                                  .bytes = bytes,
                                  .stored_bytes = load.byte_size});
        residency.complete({.resource = handle,
                            .content_generation = descriptor.content_generation,
                            .kind = texture_subresource_kind::mip,
                            .mip = load.mip,
                            .gpu_bytes = load.byte_size,
                            .succeeded = true});
    }
    CHECK(residency.resident(handle, descriptor.content_generation, texture_subresource_kind::mip, 2));
    CHECK(residency.resident(handle, descriptor.content_generation, texture_subresource_kind::mip, 3));

    const texture_mip_feedback desired{.resource = handle,
                                       .content_generation = descriptor.content_generation,
                                       .desired_mip = 0,
                                       .screen_coverage = 1.0f};
    residency.request(std::span(&desired, 1), {});
    residency.request(std::span(&desired, 1), {});
    loads = residency.take_load_requests();
    REQUIRE(loads.size() == 2);
    CHECK(residency.snapshot().deduplicated_requests > 0);

    const texture_mip_feedback stale{
        .resource = handle, .content_generation = descriptor.content_generation + 1, .desired_mip = 0};
    residency.request(std::span(&stale, 1), {});
    CHECK(residency.snapshot().stale_requests == 1);

    const auto fine = *std::find_if(loads.begin(), loads.end(), [](const auto& load) { return load.mip == 0; });
    residency.mark_loading(fine);
    residency.mark_uploading({.resource = handle,
                              .content_generation = descriptor.content_generation,
                              .kind = texture_subresource_kind::mip,
                              .mip = 0,
                              .bytes = std::make_shared<const std::vector<std::byte>>(fine.byte_size),
                              .stored_bytes = fine.byte_size});
    residency.complete({.resource = handle,
                        .content_generation = descriptor.content_generation,
                        .kind = texture_subresource_kind::mip,
                        .mip = 0,
                        .gpu_bytes = fine.byte_size,
                        .succeeded = true});
    residency.begin_frame(2);
    const auto evictions = residency.take_evictions();
    REQUIRE_FALSE(evictions.empty());
    CHECK(evictions.front().mip == 0);
    CHECK(residency.snapshot().over_budget == false);
}

TEST_CASE("texture streaming resolves backend capabilities and projected mip demand")
{
    using namespace arc::render;
    CHECK(resolve_texture_streaming_mode(texture_streaming_mode::virtual_tiles, {.mip_streaming = true}) ==
          texture_streaming_mode::streamed_mips);
    CHECK(resolve_texture_streaming_mode(texture_streaming_mode::virtual_tiles, {.mip_streaming = false}) ==
          texture_streaming_mode::resident);
    CHECK(resolve_texture_streaming_mode(texture_streaming_mode::virtual_tiles,
                                         {.mip_streaming = true, .virtual_textures = true}) ==
          texture_streaming_mode::virtual_tiles);
    CHECK(texture_requested_mip(2048, 2048, 12, 2048.0f) == 0);
    CHECK(texture_requested_mip(2048, 2048, 12, 512.0f) == 2);
    CHECK(texture_requested_mip(2048, 2048, 12, 512.0f, 1.0f) == 3);
}

TEST_CASE("texture residency exposes resolved mode and supports forced mip debugging")
{
    using namespace arc::render;
    texture_residency_manager residency({}, {.mip_streaming = true, .virtual_textures = false});
    auto descriptor = make_streamed_descriptor();
    descriptor.mode = texture_streaming_mode::virtual_tiles;
    const texture_handle handle{8, 1};
    residency.register_resource(handle, descriptor);
    auto resources = residency.resource_snapshots();
    REQUIRE(resources.size() == 1);
    CHECK(resources[0].authored_mode == texture_streaming_mode::virtual_tiles);
    CHECK(resources[0].resolved_mode == texture_streaming_mode::streamed_mips);
    CHECK(resources[0].tail_first_mip == 2);

    residency.set_forced_mip(handle, descriptor.content_generation, 1);
    resources = residency.resource_snapshots();
    REQUIRE(resources[0].forced_mip.has_value());
    CHECK(*resources[0].forced_mip == 1);
    const auto loads = residency.take_load_requests();
    CHECK(std::any_of(loads.begin(), loads.end(), [](const auto& load) { return load.mip == 1; }));
}

TEST_CASE("resident texture mode pins and requests the complete mip chain")
{
    using namespace arc::render;
    texture_residency_manager residency;
    auto descriptor = make_streamed_descriptor();
    descriptor.mode = texture_streaming_mode::resident;
    residency.register_resource({9, 1}, descriptor);
    const auto loads = residency.take_load_requests();
    REQUIRE(loads.size() == descriptor.artifact.mips.size());
    CHECK(loads.front().mip == descriptor.artifact.mips.size() - 1u);
    CHECK(loads.back().mip == 0);
}

TEST_CASE("texture residency only claims requests accepted by an IO capacity window")
{
    using namespace arc::render;
    texture_residency_manager residency;
    const texture_handle handle{12, 1};
    const auto descriptor = make_streamed_descriptor();
    residency.register_resource(handle, descriptor);

    const auto first = residency.take_load_requests(1);
    REQUIRE(first.size() == 1);
    residency.mark_loading(first.front());
    const auto second = residency.take_load_requests(1);
    REQUIRE(second.size() == 1);
    CHECK(second.front().mip != first.front().mip);
}

TEST_CASE("virtual texture page tables resolve the closest generation-valid ancestor")
{
    using namespace arc::render;
    std::vector<virtual_texture_page_table_entry> pages(3);
    pages[0] = {.generation = 7, .parent_page = 1, .flags = virtual_texture_page_flag::none, .mip = 0, .x = 1, .y = 1};
    pages[1] = {.cache_descriptor = 2,
                .cache_layer = 11,
                .generation = 7,
                .parent_page = 2,
                .flags = virtual_texture_page_flag::resident,
                .mip = 1};
    pages[2] = {.cache_descriptor = 2,
                .cache_layer = 12,
                .generation = 6,
                .parent_page = resource_handle::invalid_index,
                .flags = virtual_texture_page_flag::resident,
                .mip = 2};
    REQUIRE(resolve_virtual_texture_page(pages, 0, 7) == 1u);
    CHECK_FALSE(resolve_virtual_texture_page(pages, 0, 8).has_value());
    pages[1].flags = virtual_texture_page_flag::none;
    CHECK_FALSE(resolve_virtual_texture_page(pages, 0, 7).has_value());
}
