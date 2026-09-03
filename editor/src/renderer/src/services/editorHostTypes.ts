export type Vec3 = {
  x: number;
  y: number;
  z: number;
};

export type Transform = {
  position: Vec3;
  rotation: Vec3;
  scale: Vec3;
};

export type SceneEntity = {
  id: string;
  guid?: string;
  parentId?: string;
  name: string;
  kind: 'camera' | 'light' | 'environment' | 'mesh' | 'volume' | 'folder';
  active: boolean;
  children?: SceneEntity[];
  components?: string[];
  transform?: Transform;
  documentGuid?: string;
  editorFolder?: string;
  collection?: string;
  layer?: string;
  locked?: boolean;
  visible?: boolean;
  pickable?: boolean;
  prefabOverrideCount?: number;
};

export type AssetItem = {
  id: string;
  name: string;
  path: string;
  scope?: 'builtin' | 'project' | 'user' | 'organization';
  readOnly?: boolean;
  kind: 'scene' | 'mesh' | 'material' | 'texture' | 'environment' | 'shader' | 'prefab' | 'folder' | 'unknown';
  status: 'unknown' | 'queued' | 'ready' | 'dirty' | 'stale' | 'importing' | 'failed' | 'missing';
  guid?: string;
  typeId?: string;
  importerId?: string;
  residency?: 'metadata' | 'source' | 'derived' | 'cpu' | 'device';
  generation?: number;
  diagnostic?: string;
  dependencies?: string[];
  reverseDependencies?: string[];
  sourceBytes?: number;
  width?: number;
  height?: number;
  depth?: number;
  mipLevels?: number;
  textureFormat?: string;
  tileCount?: number;
  streamingMode?: 'resident' | 'streamed_mips' | 'virtual_tiles';
  settingsVersion?: number;
  artifactSize?: number;
  streamingEligibilityError?: string;
  vertexCount?: number;
  triangleCount?: number;
  meshCount?: number;
  materialSlotCount?: number;
  nodeCount?: number;
  animationCount?: number;
  lodCount?: number;
  entityCount?: number;
  componentCount?: number;
  nestedPrefabCount?: number;
  rootEntityName?: string;
  cameraCount?: number;
  lightCount?: number;
  materialShader?: string;
  materialParameterCount?: number;
  materialTextureCount?: number;
  shaderStages?: string[];
  shaderEntryPoints?: string[];
  shaderCompileStatus?: string;
  shaderVariantCount?: number;
  itemCount?: number;
};

export type ConsoleEvent = {
  id: string;
  level: 'info' | 'warning' | 'error' | 'debug';
  source: string;
  message: string;
  timestamp: string;
};

export type RenderStats = {
  fps: number;
  frameTimeMs: number;
  drawCalls: number;
  triangles: number;
  visibleEntities: number;
  lights: number;
  gpuMemoryMb: number;
};

export type ProjectSnapshot = {
  name: string;
  root: string;
  assetRoot: string;
  activeScene: string;
  scene: SceneEntity[];
  assets: AssetItem[];
  console: ConsoleEvent[];
  renderStats: RenderStats;
};

export const flattenScene = (entities: SceneEntity[]): SceneEntity[] =>
  entities.flatMap((entity) => [entity, ...flattenScene(entity.children ?? [])]);
