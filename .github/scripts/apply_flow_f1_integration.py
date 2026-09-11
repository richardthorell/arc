from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file = Path(path)
    text = file.read_text()
    if old not in text:
        raise RuntimeError(f"expected snippet not found in {path}: {old[:100]!r}")
    if text.count(old) != 1:
        raise RuntimeError(f"expected one snippet in {path}, found {text.count(old)}")
    file.write_text(text.replace(old, new, 1))


content = "editor/src/renderer/src/content/ContentBrowserPanel.tsx"
replace_once(content, "type CreateKind = 'material' | 'shader';", "type CreateKind = 'material' | 'flow' | 'shader';")
replace_once(
    content,
    "  { value: 'material', label: 'Material' },\n  { value: 'texture', label: 'Texture' },",
    "  { value: 'material', label: 'Material' },\n  { value: 'flow', label: 'Flow Graph' },\n  { value: 'texture', label: 'Texture' },",
)
replace_once(
    content,
    "    setCreateName(nextKind === 'material' ? 'New Material' : 'New Shader');",
    "    setCreateName(nextKind === 'material' ? 'New Material' : nextKind === 'flow' ? 'New Flow' : 'New Shader');",
)
replace_once(
    content,
    """      const request: AssetCreationRequest =
        createKind === 'material'
          ? { kind: 'material', name: createName, folder: createFolder }
          : { kind: 'shader', name: createName, folder: createFolder, template: shaderTemplate };""",
    """      const request: AssetCreationRequest =
        createKind === 'material'
          ? { kind: 'material', name: createName, folder: createFolder }
          : createKind === 'flow'
            ? { kind: 'flow', name: createName, folder: createFolder }
            : { kind: 'shader', name: createName, folder: createFolder, template: shaderTemplate };""",
)
replace_once(
    content,
    """      <button role=\"menuitem\" onClick={() => beginCreate('shader', targetFolder)}>
        <span className=\"content-create-type-icon shader\" aria-hidden=\"true\">
          {'</>'}
        </span>""",
    """      <button role=\"menuitem\" onClick={() => beginCreate('flow', targetFolder)}>
        <span className=\"content-create-type-icon shader\" aria-hidden=\"true\">
          {'⇢'}
        </span>
        <span>
          <strong>Flow Graph</strong>
          <small>Gameplay logic graph</small>
        </span>
      </button>
      <button role=\"menuitem\" onClick={() => beginCreate('shader', targetFolder)}>
        <span className=\"content-create-type-icon shader\" aria-hidden=\"true\">
          {'</>'}
        </span>""",
)
replace_once(
    content,
    "<strong id=\"content-create-title\">Create {createKind === 'material' ? 'Material' : 'Shader'}</strong>",
    "<strong id=\"content-create-title\">Create {createKind === 'material' ? 'Material' : createKind === 'flow' ? 'Flow Graph' : 'Shader'}</strong>",
)
replace_once(
    content,
    "{creating ? 'Creating…' : `Create ${createKind === 'material' ? 'Material' : 'Shader'}`}",
    "{creating ? 'Creating…' : `Create ${createKind === 'material' ? 'Material' : createKind === 'flow' ? 'Flow Graph' : 'Shader'}`}",
)

native = "editor/native/src/arc_host_base.inc"
replace_once(
    native,
    """            const bool material_asset_path = is_material_asset_path(iterator->path());
            const bool prefab_asset_path = extension == \".arcprefab\";
            if (!texture_asset && !material_asset_path && !prefab_asset_path) continue;""",
    """            const bool material_asset_path = is_material_asset_path(iterator->path());
            const bool flow_asset_path = extension == \".arcflow\";
            const bool prefab_asset_path = extension == \".arcprefab\";
            if (!texture_asset && !material_asset_path && !flow_asset_path && !prefab_asset_path) continue;""",
)
replace_once(
    native,
    """            snapshot.assets.push_back({.path = relative_path,
                                       .kind = prefab_asset_path     ? \"prefab\"
                                               : material_asset_path ? \"material\"
                                               : extension == \".hdr\" ? \"environment\"
                                                                     : \"texture\",""",
    """            snapshot.assets.push_back({.path = relative_path,
                                       .kind = prefab_asset_path     ? \"prefab\"
                                               : material_asset_path ? \"material\"
                                               : flow_asset_path     ? \"flow\"
                                               : extension == \".hdr\" ? \"environment\"
                                                                     : \"texture\",""",
)

print("Applied Flow F1 content-browser and native discovery integration")
