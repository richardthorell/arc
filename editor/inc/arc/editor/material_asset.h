#pragma once

#include <arc/render/material.h>

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
};

struct material_asset
{
    int version{ 1 };
    std::filesystem::path path;
    std::string name{ "New Material" };
    std::string shader{ "arc/default_phong" };
    std::string domain{ "surface" };
    render::material_desc material;
    material_texture_paths textures;
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
