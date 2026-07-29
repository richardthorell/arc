param(
    [string]$Glslc = "glslc",
    [string]$SpirvVal = "spirv-val",
    [string]$SourceRoot = (Join-Path $PSScriptRoot "..\assets\shaders"),
    [string]$Output = (Join-Path $PSScriptRoot "..\engine\render-vulkan\src\builtin_shaders.h")
)

$ErrorActionPreference = "Stop"
$OutputDirectory = Join-Path $PSScriptRoot "..\out\generated\shaders"
& python3 (Join-Path $PSScriptRoot "compile_builtin_shaders.py") `
    --source-root $SourceRoot `
    --output-dir $OutputDirectory `
    --header $Output `
    --glslc $Glslc `
    --spirv-val $SpirvVal `
    --write
if ($LASTEXITCODE -ne 0) { throw "ARC built-in shader generation failed" }
