from pathlib import Path

# Extend the renderer-side asset contract to match native asset kinds and the
# user-facing metadata the hover surface can consume as backends expose it.
types = Path('editor/src/renderer/src/services/editorHostTypes.ts')
text = types.read_text()
text = text.replace(
"  kind: 'scene' | 'mesh' | 'material' | 'texture' | 'shader' | 'prefab' | 'folder';",
"  kind: 'scene' | 'mesh' | 'material' | 'texture' | 'environment' | 'shader' | 'prefab' | 'folder' | 'unknown';",
)
anchor = "  triangleCount?: number;\n"
addition = """  triangleCount?: number;\n  meshCount?: number;\n  materialSlotCount?: number;\n  nodeCount?: number;\n  animationCount?: number;\n  lodCount?: number;\n  entityCount?: number;\n  componentCount?: number;\n  nestedPrefabCount?: number;\n  rootEntityName?: string;\n  cameraCount?: number;\n  lightCount?: number;\n  materialShader?: string;\n  materialParameterCount?: number;\n  materialTextureCount?: number;\n  shaderStages?: string[];\n  shaderEntryPoints?: string[];\n  shaderCompileStatus?: string;\n  shaderVariantCount?: number;\n  itemCount?: number;\n"""
if addition not in text:
    if anchor not in text:
        raise SystemExit('AssetItem field anchor not found')
    text = text.replace(anchor, addition, 1)
types.write_text(text)

card = Path('editor/src/renderer/src/content/ContentAssetCard.tsx')
text = card.read_text()
text = text.replace(
"import { assetDragType, assetPresentationIcon, assetPresentationLabel } from './assetPresentation';",
"import { assetDragType, assetPresentationIcon, assetPresentationKind, assetPresentationLabel } from './assetPresentation';",
)
start = text.index("const formatCount = (value: number | undefined) =>")
end = text.index("\nfunction AssetDetailRow", start)
replacement = r'''const formatCount = (value: number | undefined) =>
  value === undefined || !Number.isFinite(value) ? null : Math.max(0, Math.round(value)).toLocaleString();

const positiveCount = (value: number | undefined) =>
  value !== undefined && Number.isFinite(value) && value > 0 ? Math.round(value).toLocaleString() : null;

export type AssetHoverDetail = { label: string; value: string };

const pushCount = (rows: AssetHoverDetail[], label: string, value: number | undefined, includeZero = false) => {
  const formatted = includeZero ? formatCount(value) : positiveCount(value);
  if (formatted) rows.push({ label, value: formatted });
};

const pushReferences = (rows: AssetHoverDetail[], asset: AssetItem) => {
  if (asset.dependencies?.length) rows.push({ label: 'Referenced assets', value: asset.dependencies.length.toLocaleString() });
};

export const assetSpecificHoverDetails = (asset: AssetItem): AssetHoverDetail[] => {
  const rows: AssetHoverDetail[] = [];
  const kind = assetPresentationKind(asset);
  const dimensions =
    asset.width !== undefined &&
    asset.height !== undefined &&
    asset.width > 0 &&
    asset.height > 0
      ? `${asset.width} × ${asset.height}${asset.depth && asset.depth > 1 ? ` × ${asset.depth}` : ''}`
      : null;

  switch (kind) {
    case 'texture':
      if (dimensions) rows.push({ label: 'Resolution', value: dimensions });
      if (asset.textureFormat) rows.push({ label: 'Format', value: asset.textureFormat });
      if (asset.mipLevels !== undefined && asset.mipLevels > 0)
        rows.push({ label: 'Mip levels', value: String(asset.mipLevels) });
      if (asset.streamingMode)
        rows.push({
          label: 'Streaming',
          value:
            asset.streamingMode === 'streamed_mips'
              ? 'Streamed mips'
              : asset.streamingMode === 'virtual_tiles'
                ? 'Virtual texture'
                : 'Resident',
        });
      pushCount(rows, 'Virtual tiles', asset.tileCount);
      if (asset.streamingEligibilityError) rows.push({ label: 'Streaming note', value: asset.streamingEligibilityError });
      break;
    case 'environment':
      if (dimensions) rows.push({ label: 'Resolution', value: dimensions });
      if (asset.textureFormat) rows.push({ label: 'Format', value: asset.textureFormat });
      if (asset.mipLevels !== undefined && asset.mipLevels > 0)
        rows.push({ label: 'Mip levels', value: String(asset.mipLevels) });
      break;
    case 'model':
      pushCount(rows, 'Meshes', asset.meshCount);
      pushCount(rows, 'Vertices', asset.vertexCount);
      pushCount(rows, 'Triangles', asset.triangleCount);
      pushCount(rows, 'Material slots', asset.materialSlotCount);
      pushCount(rows, 'Nodes', asset.nodeCount);
      pushCount(rows, 'Animations', asset.animationCount, true);
      break;
    case 'mesh':
      pushCount(rows, 'Vertices', asset.vertexCount);
      pushCount(rows, 'Triangles', asset.triangleCount);
      pushCount(rows, 'Material slots', asset.materialSlotCount);
      pushCount(rows, 'LODs', asset.lodCount);
      break;
    case 'material':
      if (asset.materialShader) rows.push({ label: 'Shader', value: asset.materialShader });
      pushCount(rows, 'Parameters', asset.materialParameterCount, true);
      pushCount(rows, 'Textures', asset.materialTextureCount, true);
      pushReferences(rows, asset);
      break;
    case 'shader':
      if (asset.shaderStages?.length) rows.push({ label: 'Stages', value: asset.shaderStages.join(', ') });
      if (asset.shaderEntryPoints?.length) rows.push({ label: 'Entry points', value: asset.shaderEntryPoints.join(', ') });
      if (asset.shaderCompileStatus) rows.push({ label: 'Compile status', value: asset.shaderCompileStatus });
      pushCount(rows, 'Variants', asset.shaderVariantCount, true);
      break;
    case 'prefab':
      pushCount(rows, 'Entities', asset.entityCount, true);
      pushCount(rows, 'Components', asset.componentCount, true);
      pushCount(rows, 'Nested prefabs', asset.nestedPrefabCount, true);
      if (asset.rootEntityName) rows.push({ label: 'Root', value: asset.rootEntityName });
      pushReferences(rows, asset);
      break;
    case 'scene':
      pushCount(rows, 'Entities', asset.entityCount, true);
      pushCount(rows, 'Meshes', asset.meshCount, true);
      pushCount(rows, 'Cameras', asset.cameraCount, true);
      pushCount(rows, 'Lights', asset.lightCount, true);
      pushReferences(rows, asset);
      break;
    case 'folder':
      pushCount(rows, 'Items', asset.itemCount, true);
      break;
    case 'unknown':
      break;
  }

  return rows;
};
'''
text = text[:start] + replacement + text[end:]

start = text.index("function AssetHoverDetails({ asset }: { asset: AssetItem }) {")
end = text.index("\nexport function ContentAssetCard", start)
replacement = r'''function AssetHoverDetails({ asset }: { asset: AssetItem }) {
  const extension = assetFileExtension(asset);
  const specificRows = assetSpecificHoverDetails(asset);

  return (
    <UiFloatingSurface className="content-asset-hover" role="tooltip" width={TOOLTIP_WIDTH}>
      <header>
        <strong>{assetDisplayName(asset)}</strong>
        <span>{assetPresentationLabel(asset)}</span>
      </header>
      <div className="content-asset-hover-section">
        <AssetDetailRow label="Size" value={formatBytes(asset.sourceBytes)} />
        {asset.artifactSize !== undefined && asset.artifactSize > 0 && (
          <AssetDetailRow label="Cooked size" value={formatBytes(asset.artifactSize)} />
        )}
        <AssetDetailRow label="Extension" value={extension ? `.${extension}` : 'None'} />
        <AssetDetailRow label="Status" value={asset.status} />
        <AssetDetailRow label="Path" value={asset.path} />
        {asset.diagnostic && <AssetDetailRow label="Issue" value={asset.diagnostic} />}
      </div>
      {specificRows.length > 0 && (
        <div className="content-asset-hover-section asset-specific">
          {specificRows.map((row) => (
            <AssetDetailRow key={row.label} label={row.label} value={row.value} />
          ))}
        </div>
      )}
    </UiFloatingSurface>
  );
}
'''
text = text[:start] + replacement + text[end:]
card.write_text(text)

# Replace focused tests with explicit type-gating coverage and meaningful
# user-facing rows for all currently exposed asset types.
test = Path('editor/src/renderer/src/content/ContentAssetCard.test.tsx')
text = test.read_text()
text = text.replace(
"import { assetDisplayName, assetFileExtension, assetHoverPosition, ContentAssetCard } from './ContentAssetCard';",
"import {\n  assetDisplayName,\n  assetFileExtension,\n  assetHoverPosition,\n  assetSpecificHoverDetails,\n  ContentAssetCard,\n} from './ContentAssetCard';",
)
insert_anchor = "const renderCard = (asset: AssetItem) =>\n"
fixtures = r'''const materialWithTextureDefaults: AssetItem = {
  id: 'material',
  name: 'M_Wood.arcmat',
  path: 'Content/Materials/M_Wood.arcmat',
  kind: 'material',
  status: 'ready',
  width: 0,
  height: 0,
  mipLevels: 0,
  streamingMode: 'resident',
  materialShader: 'default_phong',
  materialParameterCount: 6,
  materialTextureCount: 2,
};

const model: AssetItem = {
  id: 'model',
  name: 'Cabin.glb',
  path: 'Content/Models/Cabin.glb',
  kind: 'scene',
  status: 'ready',
  width: 0,
  height: 0,
  vertexCount: 18432,
  triangleCount: 12288,
  meshCount: 3,
  materialSlotCount: 2,
  nodeCount: 5,
  animationCount: 0,
};

'''
if fixtures not in text:
    text = text.replace(insert_anchor, fixtures + insert_anchor, 1)
old = r'''  it('shows texture dimensions and mip levels when registry metadata is available', async () => {
    const view = renderCard(texture);
    const tooltip = await revealTooltip(view);

    expect(tooltip).toHaveTextContent('2048 × 1024');
    expect(tooltip).toHaveTextContent('12');
  });
'''
new = r'''  it('shows texture metadata only for texture-like asset types', async () => {
    const textureView = renderCard(texture);
    const textureTooltip = await revealTooltip(textureView);

    expect(textureTooltip).toHaveTextContent('2048 × 1024');
    expect(textureTooltip).toHaveTextContent('12');
    cleanup();
    vi.useRealTimers();

    const materialView = renderCard(materialWithTextureDefaults);
    const materialTooltip = await revealTooltip(materialView);
    expect(materialTooltip).not.toHaveTextContent('Resolution');
    expect(materialTooltip).not.toHaveTextContent('Mip levels');
    expect(materialTooltip).not.toHaveTextContent('Streaming');
    expect(materialTooltip).toHaveTextContent('default_phong');
  });

  it('defines useful per-type details for every current asset presentation kind', () => {
    expect(assetSpecificHoverDetails(model)).toEqual([
      { label: 'Meshes', value: '3' },
      { label: 'Vertices', value: '18,432' },
      { label: 'Triangles', value: '12,288' },
      { label: 'Material slots', value: '2' },
      { label: 'Nodes', value: '5' },
      { label: 'Animations', value: '0' },
    ]);
    expect(
      assetSpecificHoverDetails({
        id: 'environment',
        name: 'Studio.hdr',
        path: 'Content/Environment/Studio.hdr',
        kind: 'environment',
        status: 'ready',
        width: 4096,
        height: 2048,
        textureFormat: 'RGBA16F',
        mipLevels: 13,
      }),
    ).toEqual([
      { label: 'Resolution', value: '4096 × 2048' },
      { label: 'Format', value: 'RGBA16F' },
      { label: 'Mip levels', value: '13' },
    ]);
    expect(
      assetSpecificHoverDetails({
        id: 'shader',
        name: 'Surface.arcshader',
        path: 'Content/Shaders/Surface.arcshader',
        kind: 'shader',
        status: 'ready',
        shaderStages: ['Vertex', 'Fragment'],
        shaderEntryPoints: ['vs_main', 'fs_main'],
        shaderCompileStatus: 'Compiled',
        shaderVariantCount: 4,
      }),
    ).toHaveLength(4);
    expect(
      assetSpecificHoverDetails({
        id: 'prefab',
        name: 'Cabin.arcprefab',
        path: 'Content/Prefabs/Cabin.arcprefab',
        kind: 'prefab',
        status: 'ready',
        entityCount: 8,
        componentCount: 21,
        nestedPrefabCount: 1,
        rootEntityName: 'Cabin',
      }),
    ).toHaveLength(4);
    expect(
      assetSpecificHoverDetails({
        id: 'scene',
        name: 'Village.arcscene',
        path: 'Content/Scenes/Village.arcscene',
        kind: 'scene',
        status: 'ready',
        entityCount: 42,
        meshCount: 12,
        cameraCount: 2,
        lightCount: 5,
      }),
    ).toHaveLength(4);
    expect(
      assetSpecificHoverDetails({
        id: 'folder',
        name: 'Props',
        path: 'Content/Props',
        kind: 'folder',
        status: 'ready',
        itemCount: 17,
      }),
    ).toEqual([{ label: 'Items', value: '17' }]);
    expect(
      assetSpecificHoverDetails({
        id: 'unknown',
        name: 'Data.bin',
        path: 'Content/Data.bin',
        kind: 'unknown',
        status: 'ready',
        width: 2048,
        height: 2048,
      }),
    ).toEqual([]);
  });

  it('keeps engine implementation metadata out of the user-facing hover', async () => {
    const view = renderCard({
      ...mesh,
      importerId: 'ufbx-importer-v2',
      residency: 'device',
      readOnly: true,
    });
    const tooltip = await revealTooltip(view);

    expect(tooltip).not.toHaveTextContent('Importer');
    expect(tooltip).not.toHaveTextContent('ufbx-importer-v2');
    expect(tooltip).not.toHaveTextContent('Residency');
    expect(tooltip).not.toHaveTextContent('Engine · Read-only');
  });
'''
if old not in text:
    raise SystemExit('texture test anchor not found')
text = text.replace(old, new, 1)
test.write_text(text)
