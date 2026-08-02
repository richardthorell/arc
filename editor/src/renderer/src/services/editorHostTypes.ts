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
};

export type AssetItem = {
  id: string;
  name: string;
  path: string;
  kind: 'scene' | 'mesh' | 'material' | 'texture' | 'shader' | 'prefab' | 'folder';
  status: 'unknown' | 'queued' | 'ready' | 'dirty' | 'stale' | 'importing' | 'failed' | 'missing';
  guid?: string;
  typeId?: string;
  importerId?: string;
  residency?: 'metadata' | 'source' | 'derived' | 'cpu' | 'device';
  generation?: number;
  diagnostic?: string;
  dependencies?: string[];
  reverseDependencies?: string[];
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
