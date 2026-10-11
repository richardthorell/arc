#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace arc::editor
{

/** Saved authoring source resolved by the asset registry, never an open document buffer. */
struct material_inspection_source
{
    std::string guid;
    std::string path;
    std::string kind;
    std::string source;
};

/** A nonempty GUID is authoritative; hints are only used for legacy path-only dependencies. */
using material_inspection_resolver =
    std::function<std::optional<material_inspection_source>(std::string_view guid, std::string_view hint)>;

/**
 * Read-only native compiler reflection. Returns the materials.describe wire contract.
 * Does not allocate renderer resources, apply dirty documents, or write asset files.
 * The revision covers the saved source and resolved dependency contents/identities.
 */
[[nodiscard]] std::string inspect_saved_material(const material_inspection_source& source,
                                                 const material_inspection_resolver& resolve);

} // namespace arc::editor
