#include <arc/render_tools/render_tools.h>

namespace arc::render::tools
{

material_graph_lowering_result lower_material_graph_json(std::string_view graph_json)
{
    auto compilation = compile_material_graph_json(graph_json);
    if (!compilation) return material_graph_lowering_result::failure(compilation.error());
    return generate_material_slang(compilation.value());
}

} // namespace arc::render::tools
