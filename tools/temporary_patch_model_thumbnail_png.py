from pathlib import Path

host_path = Path('editor/native/src/arc_host_base.inc')
model_path = Path('editor/native/src/model_preview.cpp')

host = host_path.read_text()
if '#include <png.h>' not in host:
    host = host.replace('#include <nlohmann/json.hpp>\n', '#include <nlohmann/json.hpp>\n#include <png.h>\n', 1)

marker = 'std::vector<std::byte> texture_preview_bmp(const render::texture_data& texture, std::uint32_t max_size)\n'
helper = '''std::vector<std::byte> texture_preview_png_rgba(const render::texture_data& texture)\n{\n    if (!texture.has_pixels() || texture.width == 0u || texture.height == 0u) return {};\n    if (texture.format != render::texture_format::rgba8_unorm &&\n        texture.format != render::texture_format::rgba8_srgb)\n        return {};\n\n    png_image image{};\n    image.version = PNG_IMAGE_VERSION;\n    image.width = texture.width;\n    image.height = texture.height;\n    image.format = PNG_FORMAT_RGBA;\n\n    png_alloc_size_t byte_count{};\n    if (!png_image_write_to_memory(&image, nullptr, &byte_count, 0, texture.pixels.data(), 0, nullptr)) return {};\n    std::vector<std::byte> bytes(static_cast<std::size_t>(byte_count));\n    if (!png_image_write_to_memory(&image, bytes.data(), &byte_count, 0, texture.pixels.data(), 0, nullptr)) return {};\n    bytes.resize(static_cast<std::size_t>(byte_count));\n    return bytes;\n}\n\n'''
if 'texture_preview_png_rgba' not in host:
    if marker not in host:
        raise SystemExit('BMP marker not found')
    host = host.replace(marker, helper + marker, 1)

anchor = 'auto rendered = render_model_preview(imported, preview_options);'
pos = host.find(anchor)
if pos < 0:
    raise SystemExit('model preview anchor not found')
end = host.find('return snapshot;', pos)
if end < 0:
    raise SystemExit('model preview return not found')
segment = host[pos:end]
old = '''const auto bmp = texture_preview_bmp(preview, requested_size);\n        if (bmp.empty()) return std::nullopt;'''
new = '''const auto png = texture_preview_png_rgba(preview);\n        if (png.empty()) return std::nullopt;'''
if old not in segment:
    raise SystemExit('model BMP encode block not found')
segment = segment.replace(old, new, 1)
segment = segment.replace('"data:image/bmp;base64," + base64_encode(bmp)', '"data:image/png;base64," + base64_encode(png)', 1)
host = host[:pos] + segment + host[end:]
host_path.write_text(host)

model = model_path.read_text()
if 'radius * focal * 1.34f' not in model:
    raise SystemExit('camera distance constant not found')
model = model.replace('radius * focal * 1.34f', 'radius * focal * 1.12f', 1)
model_path.write_text(model)
