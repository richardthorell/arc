from pathlib import Path

metadata = Path('engine/assets/src/common/asset_metadata.cpp')
text = metadata.read_text()
if 'extension == ".obj"' not in text:
    marker = '    if (extension == ".fbx") return std::pair{asset_types::imported_scene, importer_ids::fbx};\n'
    if marker not in text:
        raise RuntimeError('FBX metadata classifier marker missing')
    text = text.replace(
        marker,
        marker + '    if (extension == ".obj") return std::pair{asset_types::imported_scene, importer_ids::obj};\n',
        1,
    )
    metadata.write_text(text)

manager = Path('engine/assets/src/common/asset_manager.cpp')
text = manager.read_text()
if 'Wavefront OBJ' not in text:
    marker = (
        '    result.push_back(std::make_unique<source_blob_importer>(importer_ids::fbx, asset_types::imported_scene, "FBX",\n'
        '                                                            std::vector<std::string>{".fbx"}));\n'
    )
    if marker not in text:
        raise RuntimeError('FBX default importer marker missing')
    addition = (
        '    result.push_back(std::make_unique<source_blob_importer>(importer_ids::obj, asset_types::imported_scene,\n'
        '                                                            "Wavefront OBJ", std::vector<std::string>{".obj"}));\n'
    )
    manager.write_text(text.replace(marker, marker + addition, 1))

required = {
    'engine/assets/inc/arc/assets/assets.h': 'asset_importer_id obj',
    'engine/assets/src/common/asset_metadata.cpp': 'extension == ".obj"',
    'engine/assets/src/common/asset_manager.cpp': 'Wavefront OBJ',
    'engine/render/src/common/mesh.cpp': 'load_obj_mesh',
    'editor/src/preload/preload.ts': "waitForImportedAsset(imported.path, 'model')",
}
for file_name, token in required.items():
    if token not in Path(file_name).read_text():
        raise RuntimeError(f'{token} missing from {file_name}')
