#pragma once

#include <arc/render/material.h>

#include <array>
#include <filesystem>
#include <string>

namespace arc::editor
{

struct material_texture_paths
{
    std::string base_color;
    std::string metallic_roughness;
    std::string normal;
    std::string ao;
    std::string emissive;
    std::string height;
    std::string clear_coat;
    std::string clear_coat_roughness;
    std::string clear_coat_normal;
    std::string anisotropy;
    std::string subsurface;
    std::string thickness;
    std::string transmission;
};

struct terrain_layer_texture_paths
{
    std::string base_color;
    std::string normal;
    std::string roughness;
    std::string ao;
    std::string height;
    // Optional prepacked override: R=AO, G=roughness, B=height.
    std::string packed_aorh;
};

struct material_asset
{
    int version{ 3 };
    std::filesystem::path path;
    std::string name{ "New Material" };
    std::string shader{ "arc/default_phong" };
    std::string domain{ "surface" };
    render::material_desc material;
    material_texture_paths textures;
    std::array<terrain_layer_texture_paths, 4> terrain_layers;
    bool graph_reserved{};
};

material_asset make_default_material_asset(std::string name = "New Material");

bool load_material_asset(
    const std::filesystem::path& path,
    const std::filesystem::path& asset_root,
    material_asset& out_asset,
    std::string& message);

bool save_material_asset(
    const material_asset& asset,
    const std::filesystem::path& asset_root,
    std::string& message);

std::filesystem::path resolve_material_texture_path(
    const std::filesystem::path& asset_root,
    const std::string& relative_path);

} // namespace arc::editor
