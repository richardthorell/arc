#include <arc/render/render.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
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
    for (std::uint32_t width = size, height = size;; width = std::max(1u, width / 2u),
                                                     height = std::max(1u, height / 2u))
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
        texture.mips.push_back({.width = width,
                                .height = height,
                                .offset = offset,
                                .size = texture.pixels.size() - offset});
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
    descriptor.artifact = {.schema_version = arc::render::texture_artifact_schema_version,
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
                           .mips = {{.width = 8, .height = 8, .offset = 4096, .stored_size = 256,
                                    .decoded_size = 256, .content_hash = 1},
                                    {.width = 4, .height = 4, .offset = 8192, .stored_size = 64,
                                     .decoded_size = 64, .content_hash = 2},
                                    {.width = 2, .height = 2, .offset = 12288, .stored_size = 16,
                                     .decoded_size = 16, .content_hash = 3},
                                    {.width = 1, .height = 1, .offset = 16384, .stored_size = 4,
                                     .decoded_size = 4, .content_hash = 4}}};
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
    for (const auto& mip : index.mips) CHECK(mip.offset % texture_artifact_alignment == 0);
    for (const auto& tile : index.tiles) CHECK(tile.offset % texture_artifact_alignment == 0);

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
    CHECK(loads[0].mip == 2);
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

    const texture_mip_feedback stale{.resource = handle,
                                     .content_generation = descriptor.content_generation + 1,
                                     .desired_mip = 0};
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
