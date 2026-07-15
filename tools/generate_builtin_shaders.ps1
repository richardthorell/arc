param(
    [string]$Glslc = "glslc",
    [string]$SourceRoot = (Join-Path $PSScriptRoot "..\assets\shaders"),
    [string]$Output = (Join-Path $PSScriptRoot "..\engine\render-vulkan\src\builtin_shaders.h")
)

$ErrorActionPreference = "Stop"
$sources = @(
    "default_phong.frag",
    "default_phong.vert",
    "default_unlit.frag",
    "default_unlit.vert",
    "deferred_lighting.frag",
    "deferred_lighting.vert",
    "gbuffer.frag",
    "gbuffer.vert",
    "shadow_depth.vert",
    "sky_atmosphere.frag",
    "sky_atmosphere.vert"
)

$temporary = Join-Path ([System.IO.Path]::GetTempPath()) "arc-builtin-shaders"
[System.IO.Directory]::CreateDirectory($temporary) | Out-Null
$builder = [System.Text.StringBuilder]::new()
[void]$builder.AppendLine("#pragma once")
[void]$builder.AppendLine()
[void]$builder.AppendLine("// Generated from assets/shaders/*.vert and *.frag with glslc.")
[void]$builder.AppendLine("#include <cstdint>")
[void]$builder.AppendLine()
[void]$builder.AppendLine("namespace arc::render::vulkan::builtin")
[void]$builder.AppendLine("{")

foreach ($source in $sources) {
    $inputPath = Join-Path $SourceRoot $source
    foreach ($line in [System.IO.File]::ReadLines($inputPath)) {
        if ($line -match '^\s*#include\s+["<]([^">]+)[">]') {
            $includePath = Join-Path $SourceRoot $Matches[1]
            if (-not (Test-Path -LiteralPath $includePath -PathType Leaf)) {
                throw "Missing shader include '$($Matches[1])' referenced by $source"
            }
        }
    }
    $spvPath = Join-Path $temporary ($source + ".spv")
    & $Glslc "--target-env=vulkan1.2" "-I$SourceRoot" $inputPath "-o" $spvPath
    if ($LASTEXITCODE -ne 0) { throw "glslc failed for $source" }
    $bytes = [System.IO.File]::ReadAllBytes($spvPath)
    if (($bytes.Length % 4) -ne 0) { throw "SPIR-V byte count is not word-aligned for $source" }
    $name = $source.Replace('.', '_') + "_spv"
    [void]$builder.AppendLine()
    [void]$builder.AppendLine("inline constexpr std::uint32_t $name[] = {")
    for ($offset = 0; $offset -lt $bytes.Length; $offset += 4) {
        $word = [System.BitConverter]::ToUInt32($bytes, $offset)
        if (($offset % 32) -eq 0) { [void]$builder.Append("    ") }
        [void]$builder.Append(("0x{0:x8}u" -f $word))
        if ($offset + 4 -lt $bytes.Length) { [void]$builder.Append(",") }
        if ((($offset + 4) % 32) -eq 0 -or $offset + 4 -eq $bytes.Length) { [void]$builder.AppendLine() }
        else { [void]$builder.Append(" ") }
    }
    [void]$builder.AppendLine("};")
}

[void]$builder.AppendLine()
[void]$builder.AppendLine("} // namespace arc::render::vulkan::builtin")
[System.IO.File]::WriteAllText($Output, $builder.ToString(), [System.Text.UTF8Encoding]::new($false))
