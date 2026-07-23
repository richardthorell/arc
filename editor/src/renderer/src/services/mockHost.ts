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
  kind: 'scene' | 'mesh' | 'material' | 'texture' | 'shader' | 'folder';
  status: 'ready' | 'dirty' | 'importing' | 'missing';
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

const transform = (position: Vec3): Transform => ({
  position,
  rotation: { x: 0, y: 0, z: 0 },
  scale: { x: 1, y: 1, z: 1 },
});

const project: ProjectSnapshot = {
  name: 'Arc Sandbox',
  root: '~/Projects/arc/sandbox',
  assetRoot: '~/Projects/arc/sandbox/assets',
  activeScene: 'scenes/demo.arcscene',
  scene: [
    {
      id: 'world',
      name: 'World',
      kind: 'folder',
      active: true,
      components: [],
      transform: transform({ x: 0, y: 0, z: 0 }),
      children: [
        {
          id: 'environment',
          name: 'World Environment',
          kind: 'environment',
          active: true,
          components: ['Sky Atmosphere', 'Height Fog', 'Environment Probe'],
          transform: transform({ x: 0, y: 0, z: 0 }),
        },
        {
          id: 'sun',
          name: 'Directional Light',
          kind: 'light',
          active: true,
          components: ['Directional Light', 'Shadow Cascade'],
          transform: {
            position: { x: 0, y: 5, z: 0 },
            rotation: { x: -42, y: 35, z: 0 },
            scale: { x: 1, y: 1, z: 1 },
          },
        },
      ],
    },
    {
      id: 'camera-main',
      name: 'Main Camera',
      kind: 'camera',
      active: true,
      components: ['Camera', 'Viewport Controller'],
      transform: {
        position: { x: -4.25, y: 3.2, z: 7.5 },
        rotation: { x: -22, y: -31, z: 0 },
        scale: { x: 1, y: 1, z: 1 },
      },
    },
    {
      id: 'set-dressing',
      name: 'Set Dressing',
      kind: 'folder',
      active: true,
      components: [],
      transform: transform({ x: 0, y: 0, z: 0 }),
      children: [
        {
          id: 'prototype-cube',
          name: 'Prototype Cube',
          kind: 'mesh',
          active: true,
          components: ['Transform', 'Mesh Renderer', 'Collider'],
          transform: transform({ x: 0, y: 1, z: 0 }),
        },
        {
          id: 'floor',
          name: 'Floor Plane',
          kind: 'mesh',
          active: true,
          components: ['Transform', 'Mesh Renderer'],
          transform: {
            position: { x: 0, y: 0, z: 0 },
            rotation: { x: 0, y: 0, z: 0 },
            scale: { x: 12, y: 1, z: 12 },
          },
        },
        {
          id: 'fog-volume',
          name: 'Fog Volume',
          kind: 'volume',
          active: false,
          components: ['Transform', 'Height Fog Volume'],
          transform: {
            position: { x: 2, y: 0.5, z: -3 },
            rotation: { x: 0, y: 0, z: 0 },
            scale: { x: 6, y: 2, z: 6 },
          },
        },
      ],
    },
  ],
  assets: [
    { id: 'asset-scene-demo', name: 'demo.arcscene', path: 'scenes/demo.arcscene', kind: 'scene', status: 'dirty' },
    { id: 'asset-mesh-cube', name: 'cube.arcmesh', path: 'meshes/cube.arcmesh', kind: 'mesh', status: 'ready' },
    { id: 'asset-material-default', name: 'default.arcmat', path: 'materials/default.arcmat', kind: 'material', status: 'ready' },
    { id: 'asset-material-grid', name: 'grid_floor.arcmat', path: 'materials/grid_floor.arcmat', kind: 'material', status: 'ready' },
    { id: 'asset-texture-checker', name: 'checker.ktx', path: 'textures/checker.ktx', kind: 'texture', status: 'ready' },
    { id: 'asset-shader-pbr', name: 'pbr_lit.hlsl', path: 'shaders/pbr_lit.hlsl', kind: 'shader', status: 'ready' },
    { id: 'asset-shader-outline', name: 'selection_outline.hlsl', path: 'shaders/selection_outline.hlsl', kind: 'shader', status: 'importing' },
  ],
  console: [
    { id: 'event-1', level: 'info', source: 'editor', message: 'Workbench initialized with mock host data.', timestamp: '09:41:12' },
    { id: 'event-2', level: 'info', source: 'project', message: 'Loaded project Arc Sandbox.', timestamp: '09:41:12' },
    { id: 'event-3', level: 'debug', source: 'renderer', message: 'Viewport is running in placeholder mode.', timestamp: '09:41:13' },
    { id: 'event-4', level: 'warning', source: 'assets', message: 'selection_outline.hlsl is still importing.', timestamp: '09:41:13' },
  ],
  renderStats: {
    fps: 144,
    frameTimeMs: 6.94,
    drawCalls: 128,
    triangles: 84216,
    visibleEntities: 12,
    lights: 24,
    gpuMemoryMb: 384,
  },
};

const delay = async () => new Promise((resolve) => window.setTimeout(resolve, 80));

export const mockHost = {
  async getProjectSnapshot(): Promise<ProjectSnapshot> {
    await delay();
    return project;
  },

  async selectEntity(entityId: string): Promise<{ selectedEntityId: string }> {
    await delay();
    return { selectedEntityId: entityId };
  },

  async executeNoop(command: string): Promise<{ command: string; succeeded: boolean }> {
    await delay();
    return { command, succeeded: true };
  },
};

export const flattenScene = (entities: SceneEntity[]): SceneEntity[] =>
  entities.flatMap((entity) => [entity, ...flattenScene(entity.children ?? [])]);
