import {
  cloneMaterialGraph,
  isMaterialGraph,
  type MaterialAssetJson,
  type MaterialGraph,
  type MaterialGraphConnection,
  type MaterialGraphNode,
  type MaterialGraphNodeType,
} from './materialGraphTypes';

export const currentMaterialAuthoringVersion = 4;

export type MaterialAssetUpgradeResult = {
  asset: MaterialAssetJson;
  sourceVersion: number;
  upgraded: boolean;
};

type LegacyColor = { r?: unknown; g?: unknown; b?: unknown; a?: unknown };
type LegacySurface = {
  baseColor?: LegacyColor;
  metallic?: unknown;
  roughness?: unknown;
  normalScale?: unknown;
  aoStrength?: unknown;
  emissive?: LegacyColor;
  emissiveStrength?: unknown;
  alphaCutoff?: unknown;
};
type LegacyAdvanced = {
  clearCoat?: unknown;
  clearCoatRoughness?: unknown;
  clearCoatNormalScale?: unknown;
  sheen?: unknown;
  transmission?: unknown;
  indexOfRefraction?: unknown;
  thickness?: unknown;
  attenuationColor?: LegacyColor;
  attenuationDistance?: unknown;
  subsurface?: unknown;
  subsurfaceColor?: LegacyColor;
  anisotropy?: unknown;
  anisotropyRotation?: unknown;
  parallaxHeightScale?: unknown;
};
type LegacyTextures = {
  baseColor?: unknown;
  metallicRoughness?: unknown;
  normal?: unknown;
  ao?: unknown;
  emissive?: unknown;
  height?: unknown;
  clearCoat?: unknown;
  clearCoatRoughness?: unknown;
  clearCoatNormal?: unknown;
  anisotropy?: unknown;
  subsurface?: unknown;
  thickness?: unknown;
  transmission?: unknown;
};

const finite = (value: unknown, fallback: number) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;
const stringValue = (value: unknown) => (typeof value === 'string' ? value.trim() : '');
const recordValue = <T extends object>(value: unknown): T =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as T) : ({} as T);
const color3 = (value: LegacyColor | undefined, fallback: [number, number, number]): [number, number, number] => [
  finite(value?.r, fallback[0]),
  finite(value?.g, fallback[1]),
  finite(value?.b, fallback[2]),
];

const legacyMaterialGraph = (asset: MaterialAssetJson): MaterialGraph => {
  const surface = recordValue<LegacySurface>(asset.surface);
  const advanced = recordValue<LegacyAdvanced>(asset.advanced);
  const textures = recordValue<LegacyTextures>(asset.textures);
  const nodes: MaterialGraphNode[] = [];
  const connections: MaterialGraphConnection[] = [];
  let y = 80;
  let connectionId = 0;

  const node = (
    id: string,
    type: MaterialGraphNodeType,
    values: Record<string, unknown>,
    x = 80,
    positionY = y,
    parameterName?: string,
  ) => {
    const created: MaterialGraphNode = { id, type, position: [x, positionY], values };
    if (parameterName) created.parameter = { exposed: true, name: parameterName };
    nodes.push(created);
    y = Math.max(y, positionY + 120);
    return created;
  };
  const connect = (from: MaterialGraphNode, fromPin: string, to: MaterialGraphNode, toPin: string) => {
    connections.push({
      id: `legacy-connection-${connectionId++}`,
      from: { nodeId: from.id, pin: fromPin },
      to: { nodeId: to.id, pin: toPin },
    });
  };
  const output = node('material-output', 'output', {}, 760, 260);

  const scalar = (
    id: string,
    label: string,
    value: number,
    outputPin: string,
    defaultValue: number,
    force = false,
  ) => {
    if (!force && Math.abs(value - defaultValue) <= 1e-7) return undefined;
    const source = node(id, 'constant', { value }, 80, y, label);
    connect(source, 'value', output, outputPin);
    return source;
  };
  const vector = (
    id: string,
    label: string,
    value: [number, number, number],
    outputPin: string,
    positionY = y,
  ) => {
    const source = node(id, 'vector3', { value }, 80, positionY, label);
    connect(source, 'value', output, outputPin);
    return source;
  };
  const texture = (id: string, path: unknown, positionY: number) => {
    const source = stringValue(path);
    return source ? node(id, 'textureSample2D', { texture: source, dimension: '2d' }, 330, positionY) : undefined;
  };
  const multiply = (
    id: string,
    left: MaterialGraphNode,
    leftPin: string,
    right: MaterialGraphNode,
    rightPin: string,
    toPin: string,
    positionY: number,
  ) => {
    const product = node(id, 'multiply', {}, 560, positionY);
    connect(left, leftPin, product, 'a');
    connect(right, rightPin, product, 'b');
    connect(product, 'result', output, toPin);
    return product;
  };

  const baseColor = color3(surface.baseColor, [0.78, 0.8, 0.84]);
  const base = node('legacy-base-color', 'vector3', { value: baseColor }, 80, 80, 'Base Color');
  const baseTexture = texture('legacy-base-color-texture', textures.baseColor, 80);
  if (baseTexture) multiply('legacy-base-color-multiply', baseTexture, 'rgb', base, 'value', 'baseColor', 80);
  else connect(base, 'value', output, 'baseColor');

  const metallicValue = finite(surface.metallic, 0);
  const roughnessValue = finite(surface.roughness, 0.62);
  const metallic = node('legacy-metallic', 'constant', { value: metallicValue }, 80, 220, 'Metallic');
  const roughness = node('legacy-roughness', 'constant', { value: roughnessValue }, 80, 340, 'Roughness');
  const packedSurface = texture('legacy-metallic-roughness-texture', textures.metallicRoughness, 220);
  if (packedSurface) {
    multiply('legacy-metallic-multiply', packedSurface, 'b', metallic, 'value', 'metallic', 220);
    multiply('legacy-roughness-multiply', packedSurface, 'g', roughness, 'value', 'roughness', 340);
  } else {
    connect(metallic, 'value', output, 'metallic');
    connect(roughness, 'value', output, 'roughness');
  }

  const normalTexture = texture('legacy-normal-texture', textures.normal, 480);
  if (normalTexture) {
    const normal = node(
      'legacy-normal-map',
      'normalMap',
      { strength: finite(surface.normalScale, 1) },
      560,
      480,
    );
    connect(normalTexture, 'rgb', normal, 'texture');
    connect(normal, 'normal', output, 'normal');
  }

  const aoTexture = texture('legacy-ao-texture', textures.ao, 600);
  if (aoTexture) {
    const strength = node(
      'legacy-ao-strength',
      'constant',
      { value: finite(surface.aoStrength, 1) },
      80,
      600,
      'AO Strength',
    );
    multiply('legacy-ao-multiply', aoTexture, 'r', strength, 'value', 'ao', 600);
  } else {
    scalar('legacy-ao', 'Ambient Occlusion', finite(surface.aoStrength, 1), 'ao', 1);
  }

  const emissiveStrength = finite(surface.emissiveStrength, 0);
  const rawEmissive = color3(surface.emissive, [0, 0, 0]);
  const emissiveValue: [number, number, number] = [
    rawEmissive[0] * emissiveStrength,
    rawEmissive[1] * emissiveStrength,
    rawEmissive[2] * emissiveStrength,
  ];
  const emissive = node('legacy-emissive', 'vector3', { value: emissiveValue }, 80, 740, 'Emissive');
  const emissiveTexture = texture('legacy-emissive-texture', textures.emissive, 740);
  if (emissiveTexture)
    multiply('legacy-emissive-multiply', emissiveTexture, 'rgb', emissive, 'value', 'emissive', 740);
  else if (emissiveValue.some((component) => Math.abs(component) > 1e-7))
    connect(emissive, 'value', output, 'emissive');

  const opacity = finite(surface.baseColor?.a, 1);
  const opacityTexture = baseTexture;
  if (opacityTexture) {
    const opacityFactor = node('legacy-opacity-factor', 'constant', { value: opacity }, 80, 860, 'Opacity');
    multiply('legacy-opacity-multiply', opacityTexture, 'a', opacityFactor, 'value', 'opacity', 860);
  } else {
    scalar('legacy-opacity', 'Opacity', opacity, 'opacity', 1);
  }
  scalar('legacy-alpha-clip', 'Alpha Clip', finite(surface.alphaCutoff, 0.5), 'alphaClip', 0.5,
    String(asset.blendMode ?? '').toLowerCase() === 'masked');

  if (!stringValue(textures.clearCoat))
    scalar('legacy-clear-coat', 'Clear Coat', finite(advanced.clearCoat, 0), 'clearCoat', 0);
  if (!stringValue(textures.clearCoatRoughness))
    scalar(
      'legacy-clear-coat-roughness',
      'Clear Coat Roughness',
      finite(advanced.clearCoatRoughness, 0.1),
      'clearCoatRoughness',
      0.1,
    );
  scalar('legacy-sheen', 'Sheen', finite(advanced.sheen, 0), 'sheen', 0);
  if (!stringValue(textures.transmission))
    scalar('legacy-transmission', 'Transmission', finite(advanced.transmission, 0), 'transmission', 0);
  scalar(
    'legacy-index-of-refraction',
    'Index of Refraction',
    finite(advanced.indexOfRefraction, 1.5),
    'indexOfRefraction',
    1.5,
  );
  if (!stringValue(textures.thickness))
    scalar('legacy-thickness', 'Thickness', finite(advanced.thickness, 0), 'thickness', 0);
  scalar(
    'legacy-attenuation-distance',
    'Attenuation Distance',
    finite(advanced.attenuationDistance, 1),
    'attenuationDistance',
    1,
  );
  if (!stringValue(textures.subsurface))
    scalar('legacy-subsurface', 'Subsurface', finite(advanced.subsurface, 0), 'subsurface', 0);
  if (!stringValue(textures.anisotropy))
    scalar('legacy-anisotropy', 'Anisotropy', finite(advanced.anisotropy, 0), 'anisotropy', 0);
  scalar(
    'legacy-anisotropy-rotation',
    'Anisotropy Rotation',
    finite(advanced.anisotropyRotation, 0),
    'anisotropyRotation',
    0,
  );

  const attenuation = color3(advanced.attenuationColor, [1, 1, 1]);
  if (attenuation.some((component) => Math.abs(component - 1) > 1e-7))
    vector('legacy-attenuation-color', 'Attenuation Color', attenuation, 'attenuationColor');
  const subsurfaceColor = color3(advanced.subsurfaceColor, [1, 0.35, 0.2]);
  if (
    Math.abs(subsurfaceColor[0] - 1) > 1e-7 ||
    Math.abs(subsurfaceColor[1] - 0.35) > 1e-7 ||
    Math.abs(subsurfaceColor[2] - 0.2) > 1e-7
  )
    vector('legacy-subsurface-color', 'Subsurface Color', subsurfaceColor, 'subsurfaceColor');

  const scalarTexture = (
    id: string,
    path: unknown,
    outputPin: string,
    factor: number,
    defaultFactor: number,
    positionY: number,
  ) => {
    const sampled = texture(`${id}-texture`, path, positionY);
    if (!sampled) return;
    if (Math.abs(factor - defaultFactor) <= 1e-7) {
      connect(sampled, 'r', output, outputPin);
      return;
    }
    const multiplier = node(`${id}-factor`, 'constant', { value: factor }, 80, positionY, outputPin);
    multiply(`${id}-multiply`, sampled, 'r', multiplier, 'value', outputPin, positionY);
  };
  scalarTexture('legacy-clear-coat-texture', textures.clearCoat, 'clearCoat', finite(advanced.clearCoat, 0), 1, 980);
  scalarTexture(
    'legacy-clear-coat-roughness-texture',
    textures.clearCoatRoughness,
    'clearCoatRoughness',
    finite(advanced.clearCoatRoughness, 0.1),
    1,
    1100,
  );
  scalarTexture(
    'legacy-anisotropy-texture',
    textures.anisotropy,
    'anisotropy',
    finite(advanced.anisotropy, 0),
    1,
    1220,
  );
  scalarTexture(
    'legacy-subsurface-texture',
    textures.subsurface,
    'subsurface',
    finite(advanced.subsurface, 0),
    1,
    1340,
  );
  scalarTexture(
    'legacy-thickness-texture',
    textures.thickness,
    'thickness',
    finite(advanced.thickness, 0),
    1,
    1460,
  );
  scalarTexture(
    'legacy-transmission-texture',
    textures.transmission,
    'transmission',
    finite(advanced.transmission, 0),
    1,
    1580,
  );

  const clearCoatNormalTexture = texture('legacy-clear-coat-normal-texture', textures.clearCoatNormal, 1700);
  if (clearCoatNormalTexture) {
    const clearCoatNormal = node(
      'legacy-clear-coat-normal-map',
      'normalMap',
      { strength: finite(advanced.clearCoatNormalScale, 1) },
      560,
      1700,
    );
    connect(clearCoatNormalTexture, 'rgb', clearCoatNormal, 'texture');
    connect(clearCoatNormal, 'normal', output, 'clearCoatNormal');
  }

  return {
    version: 1,
    nodes,
    connections,
    viewport: { x: 40, y: 40, zoom: 0.85 },
  };
};

export const upgradeMaterialAsset = (asset: MaterialAssetJson): MaterialAssetUpgradeResult => {
  const rawVersion = asset.version;
  const sourceVersion = rawVersion === undefined ? 1 : rawVersion;
  if (!Number.isInteger(sourceVersion) || sourceVersion < 1 || sourceVersion > currentMaterialAuthoringVersion)
    throw new Error(`Unsupported material authoring schema v${String(sourceVersion)}`);

  if (sourceVersion === currentMaterialAuthoringVersion)
    return { asset, sourceVersion, upgraded: false };

  const shaderPath = stringValue(asset.shaderPath);
  const existingGraph = isMaterialGraph(asset.graph) ? cloneMaterialGraph(asset.graph) : undefined;
  const graph = shaderPath ? null : existingGraph ?? legacyMaterialGraph(asset);
  const upgraded: MaterialAssetJson = {
    ...asset,
    version: currentMaterialAuthoringVersion,
    graph,
  };
  if (shaderPath) upgraded.shaderPath = shaderPath;
  else delete upgraded.shaderPath;

  const legacyShader = stringValue(asset.shader);
  const legacyTextures = recordValue<LegacyTextures>(asset.textures);
  const legacyHeightTexture = stringValue(legacyTextures.height);
  const parallaxHeightScale = finite(recordValue<LegacyAdvanced>(asset.advanced).parallaxHeightScale, 0);
  if (legacyShader || legacyHeightTexture || Math.abs(parallaxHeightScale) > 1e-7) {
    upgraded.migrationMetadata = {
      ...(recordValue<Record<string, unknown>>(asset.migrationMetadata) ?? {}),
      ...(legacyShader ? { legacyShader } : {}),
      ...(legacyHeightTexture ? { legacyHeightTexture } : {}),
      ...(Math.abs(parallaxHeightScale) > 1e-7 ? { legacyParallaxHeightScale: parallaxHeightScale } : {}),
    };
  }

  delete upgraded.shader;
  delete upgraded.surface;
  delete upgraded.textures;
  delete upgraded.advanced;
  return { asset: upgraded, sourceVersion, upgraded: true };
};
