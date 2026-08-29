import { z } from 'zod';

export type HostEntityId = { index: number; generation: number };
export type Vec3 = { x: number; y: number; z: number };
export type Vec4 = { x: number; y: number; z: number; w: number };
export type Quaternion = { x: number; y: number; z: number; w: number };

export type InspectorTransform = {
  position: Vec3;
  rotationDegrees: Vec3;
  scale: Vec3;
  rotationQuaternion: Quaternion;
};

export type InspectorCamera = {
  projection: 'perspective' | 'orthographic';
  fovYDegrees: number;
  orthographicHeight: number;
  nearPlane: number;
  farPlane: number;
  active: boolean;
  clearColor: Vec4;
  exposureMode: 'manual' | 'automatic';
  exposureMetering: 'average' | 'centerWeighted';
  manualEV100: number;
  exposureCompensation: number;
  minimumEV100: number;
  maximumEV100: number;
  brightenSpeed: number;
  darkenSpeed: number;
  antiAliasing: 'inherit' | 'disabled' | 'fxaa' | 'taa' | 'taau';
};

export type InspectorLight = {
  kind: 'directional' | 'point' | 'spot' | 'rectangle' | 'disk';
  unit: 'unitless' | 'lumens' | 'candela' | 'lux' | 'nits';
  color: Vec4;
  intensity: number;
  range: number;
  innerAngleDegrees: number;
  outerAngleDegrees: number;
  width: number;
  height: number;
  twoSided: boolean;
  enabled: boolean;
  castsShadows: boolean;
  shadowResolution: number;
  shadowPriority: number;
  shadowStrength: number;
  shadowBias: number;
  shadowNormalBias: number;
  shadowFilter: number;
  contactShadows: boolean;
  contactShadowLength: number;
  shadowCacheMode: number;
  shadowMapMethod: number;
  cascadeCount: number;
  shadowDistance: number;
  cascadeSplitLambda: number;
  cascadeBlendFraction: number;
  stableCascades: boolean;
  useColorTemperature: boolean;
  temperatureKelvin: number;
};

export type InspectorMeshRenderer = {
  representation: 'auto' | 'conventional' | 'virtualized';
  visible: boolean;
  castsShadows: boolean;
  receivesShadows: boolean;
  shadowLodBias: number;
  maximumShadowDistance: number;
  baseColorTint: Vec4;
  hasMaterial: boolean;
  assetBackedMaterial: boolean;
  materialName: string;
  materialPath: string;
};

export type InspectorTerrain = {
  enabled: boolean;
  size: number;
  minimumElevation?: number;
  maximumElevation?: number;
  resolution: number;
  chunkQuads: number;
  patchQuads: number;
  maximumHierarchyDepth: number;
  geometricErrorMultiplier: number;
  receiveShadows: boolean;
  castShadows: boolean;
  shadowLodBias: number;
  maximumShadowDistance: number;
  contentRevision: number;
  materialGuid?: string;
  materialPath?: string;
  hierarchyNodes?: number;
  hierarchyDepth?: number;
  sourcePatches?: number;
  visiblePatches?: number;
  renderedTriangles?: number;
  cpuMemoryBytes?: number;
  gpuMemoryBytes?: number;
  uploadedBytes?: number;
  brushTool: 'sculpt' | 'smooth' | 'flatten' | 'paint';
  brushRadius: number;
  brushStrength: number;
  brushFalloff: number;
  activeLayer: number;
  layers: Array<{ name: string; baseColorPath: string }>;
};

export type InspectorPrefab = {
  prefabGuid: string;
  prefabPath: string;
  overrideCount: number;
  sourceMissing: boolean;
  overrides: Array<{ sourceEntity: string; componentId: string; fieldId: number; kind: string }>;
};

export type HostComponentSnapshot = {
  kind: string;
  label: string;
  editable: boolean;
};

export type InspectorProjectComponent = {
  typeId: string;
  canonicalName: string;
  displayName: string;
  schemaVersion: number;
  values: Record<string, unknown>;
};

export type InspectorEntitySnapshot = {
  entity: HostEntityId;
  selectionCount?: number;
  selectedGuids?: string[];
  name: string;
  tag: string;
  active: boolean;
  renderLayerMask: number;
  mobility?: 'static' | 'stationary' | 'movable';
  transform: InspectorTransform | null;
  camera: InspectorCamera | null;
  light: InspectorLight | null;
  meshRenderer: InspectorMeshRenderer | null;
  terrain: InspectorTerrain | null;
  prefab: InspectorPrefab | null;
  components: HostComponentSnapshot[];
  projectComponents: InspectorProjectComponent[];
  aggregate?: {
    mixedFields: string[];
    commonComponents: string[];
    partialComponents: string[];
  };
};

export type HostResponse<T = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: T;
};

const finiteNumber = z.number().finite();
const entityIdSchema = z.object({
  index: z.number().int().nonnegative().max(0xfffffffe),
  generation: z.number().int().nonnegative(),
});
const vec3Tuple = z.tuple([finiteNumber, finiteNumber, finiteNumber]);
const vec4Tuple = z.tuple([finiteNumber, finiteNumber, finiteNumber, finiteNumber]);
const quaternionTuple = z.tuple([finiteNumber, finiteNumber, finiteNumber, finiteNumber]);

const hostSelectedEntitySchema = z.object({
  entity: entityIdSchema,
  selectionCount: z.number().int().nonnegative().default(1),
  selectedGuids: z.array(z.string()).default([]),
  name: z.string(),
  tag: z.string(),
  active: z.boolean(),
  renderLayerMask: z.number().int().nonnegative(),
  mobility: z.enum(['static', 'stationary', 'movable']).default('movable'),
  transform: z
    .object({
      position: vec3Tuple,
      rotation: quaternionTuple,
      scale: vec3Tuple,
    })
    .nullable(),
  camera: z
    .object({
      projection: z.enum(['perspective', 'orthographic']),
      fovYDegrees: finiteNumber,
      orthographicHeight: finiteNumber,
      nearPlane: finiteNumber,
      farPlane: finiteNumber,
      active: z.boolean(),
      clearColor: vec4Tuple,
      exposureMode: z.enum(['manual', 'automatic']).default('automatic'),
      exposureMetering: z.enum(['average', 'centerWeighted']).default('average'),
      manualEV100: finiteNumber.default(10),
      exposureCompensation: finiteNumber.default(0),
      minimumEV100: finiteNumber.default(-8),
      maximumEV100: finiteNumber.default(20),
      brightenSpeed: finiteNumber.nonnegative().default(3),
      darkenSpeed: finiteNumber.nonnegative().default(1),
      antiAliasing: z.enum(['inherit', 'disabled', 'fxaa', 'taa', 'taau']).default('inherit'),
    })
    .nullable(),
  light: z
    .object({
      kind: z.enum(['directional', 'point', 'spot', 'rectangle', 'disk']),
      unit: z.enum(['unitless', 'lumens', 'candela', 'lux', 'nits']),
      color: vec3Tuple,
      intensity: finiteNumber.nonnegative(),
      range: finiteNumber.positive(),
      innerAngleDegrees: finiteNumber.nonnegative(),
      outerAngleDegrees: finiteNumber.positive(),
      width: finiteNumber.positive(),
      height: finiteNumber.positive(),
      twoSided: z.boolean(),
      enabled: z.boolean(),
      castsShadows: z.boolean(),
      shadowResolution: z.number().int().min(128).max(8192).default(2048),
      shadowPriority: z.number().int().min(0).max(65535).default(128),
      shadowStrength: finiteNumber.min(0).max(1).default(0.75),
      shadowBias: finiteNumber.nonnegative().default(0.0015),
      shadowNormalBias: finiteNumber.nonnegative().default(0.01),
      shadowFilter: z.number().int().min(0).max(3).default(1),
      contactShadows: z.boolean().default(true),
      contactShadowLength: finiteNumber.nonnegative().default(0.5),
      shadowCacheMode: z.number().int().min(0).max(2).default(0),
      shadowMapMethod: z.number().int().min(0).max(2).default(0),
      cascadeCount: z.number().int().min(1).max(4).default(4),
      shadowDistance: finiteNumber.positive().default(200),
      cascadeSplitLambda: finiteNumber.min(0).max(1).default(0.65),
      cascadeBlendFraction: finiteNumber.min(0).max(0.3).default(0.1),
      stableCascades: z.boolean().default(true),
      useColorTemperature: z.boolean(),
      temperatureKelvin: finiteNumber.min(1000).max(40000),
    })
    .nullable()
    .default(null),
  meshRenderer: z
    .object({
      representation: z.enum(['auto', 'conventional', 'virtualized']).default('auto'),
      visible: z.boolean(),
      castsShadows: z.boolean().default(true),
      receivesShadows: z.boolean().default(true),
      shadowLodBias: finiteNumber.min(-4).max(8).default(0),
      maximumShadowDistance: finiteNumber.nonnegative().default(0),
      baseColorTint: vec4Tuple,
      hasMaterial: z.boolean(),
      assetBackedMaterial: z.boolean(),
      materialName: z.string(),
      materialPath: z.string(),
    })
    .nullable(),
  terrain: z
    .object({
      enabled: z.boolean(),
      size: finiteNumber.positive(),
      minimumElevation: finiteNumber.default(0),
      maximumElevation: finiteNumber.default(0),
      resolution: z.number().int().min(3),
      chunkQuads: z.number().int().positive(),
      patchQuads: z.union([z.literal(16), z.literal(32), z.literal(64)]),
      maximumHierarchyDepth: z.number().int().nonnegative(),
      geometricErrorMultiplier: z.number().positive(),
      receiveShadows: z.boolean(),
      castShadows: z.boolean().default(true),
      shadowLodBias: finiteNumber.min(-4).max(8).default(0),
      maximumShadowDistance: finiteNumber.nonnegative().default(0),
      contentRevision: z.number().int().nonnegative(),
      materialGuid: z.string().default(''),
      materialPath: z.string().default(''),
      hierarchyNodes: z.number().int().nonnegative().default(0),
      hierarchyDepth: z.number().int().nonnegative().default(0),
      sourcePatches: z.number().int().nonnegative().default(0),
      visiblePatches: z.number().int().nonnegative().default(0),
      renderedTriangles: z.number().int().nonnegative().default(0),
      cpuMemoryBytes: z.number().int().nonnegative().default(0),
      gpuMemoryBytes: z.number().int().nonnegative().default(0),
      uploadedBytes: z.number().int().nonnegative().default(0),
      brushTool: z.enum(['sculpt', 'smooth', 'flatten', 'paint']),
      brushRadius: finiteNumber.min(0.25).max(128),
      brushStrength: finiteNumber.positive().max(1),
      brushFalloff: finiteNumber.min(0).max(1),
      activeLayer: z.number().int().min(0).max(3),
      layers: z.array(z.object({ name: z.string(), baseColorPath: z.string() })).length(4),
    })
    .nullable()
    .default(null),
  prefab: z
    .object({
      prefabGuid: z.string(),
      prefabPath: z.string(),
      overrideCount: z.number().int().nonnegative(),
      sourceMissing: z.boolean(),
      overrides: z
        .array(
          z.object({
            sourceEntity: z.string(),
            componentId: z.string(),
            fieldId: z.number().int().nonnegative(),
            kind: z.string(),
          }),
        )
        .default([]),
    })
    .nullable()
    .default(null),
  components: z.array(
    z.object({
      kind: z.string(),
      label: z.string(),
      editable: z.boolean(),
    }),
  ),
  projectComponents: z
    .array(
      z.object({
        typeId: z.string().length(32),
        canonicalName: z.string(),
        displayName: z.string(),
        schemaVersion: z.number().int().positive(),
        values: z.record(z.string(), z.unknown()),
      }),
    )
    .default([]),
});

const tupleToVec3 = (value: [number, number, number]): Vec3 => ({ x: value[0], y: value[1], z: value[2] });
const tupleToVec4 = (value: [number, number, number, number]): Vec4 => ({
  x: value[0],
  y: value[1],
  z: value[2],
  w: value[3],
});
const tupleToQuaternion = (value: [number, number, number, number]): Quaternion => ({
  x: value[0],
  y: value[1],
  z: value[2],
  w: value[3],
});
const radiansToDegrees = (value: number) => (value * 180) / Math.PI;
const degreesToRadians = (value: number) => (value * Math.PI) / 180;

export function quaternionToEulerDegrees(value: Quaternion): Vec3 {
  const length = Math.hypot(value.x, value.y, value.z, value.w) || 1;
  const x = value.x / length;
  const y = value.y / length;
  const z = value.z / length;
  const w = value.w / length;
  const sinY = Math.max(-1, Math.min(1, 2 * (x * z + w * y)));
  return {
    x: radiansToDegrees(Math.atan2(2 * (w * x - y * z), 1 - 2 * (x * x + y * y))),
    y: radiansToDegrees(Math.asin(sinY)),
    z: radiansToDegrees(Math.atan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z))),
  };
}

export function eulerDegreesToQuaternion(value: Vec3): Quaternion {
  const halfX = degreesToRadians(value.x) * 0.5;
  const halfY = degreesToRadians(value.y) * 0.5;
  const halfZ = degreesToRadians(value.z) * 0.5;
  const cx = Math.cos(halfX);
  const sx = Math.sin(halfX);
  const cy = Math.cos(halfY);
  const sy = Math.sin(halfY);
  const cz = Math.cos(halfZ);
  const sz = Math.sin(halfZ);
  return {
    x: sx * cy * cz + cx * sy * sz,
    y: cx * sy * cz - sx * cy * sz,
    z: cx * cy * sz + sx * sy * cz,
    w: cx * cy * cz - sx * sy * sz,
  };
}

export function parseSelectedEntitySnapshot(value: unknown): InspectorEntitySnapshot {
  const parsed = hostSelectedEntitySchema.parse(value);
  const rotationQuaternion = parsed.transform ? tupleToQuaternion(parsed.transform.rotation) : null;
  return {
    ...parsed,
    transform:
      parsed.transform && rotationQuaternion
        ? {
            position: tupleToVec3(parsed.transform.position),
            rotationDegrees: quaternionToEulerDegrees(rotationQuaternion),
            scale: tupleToVec3(parsed.transform.scale),
            rotationQuaternion,
          }
        : null,
    camera: parsed.camera
      ? {
          ...parsed.camera,
          clearColor: tupleToVec4(parsed.camera.clearColor),
        }
      : null,
    light: parsed.light
      ? {
          ...parsed.light,
          color: { ...tupleToVec3(parsed.light.color), w: 1 },
        }
      : null,
    meshRenderer: parsed.meshRenderer
      ? {
          ...parsed.meshRenderer,
          baseColorTint: tupleToVec4(parsed.meshRenderer.baseColorTint),
        }
      : null,
  };
}

const aggregateComponentKeys = ['transform', 'camera', 'light', 'meshRenderer', 'terrain', 'prefab'] as const;

const flattenSnapshotValues = (value: unknown, prefix: string, output: Map<string, string>): void => {
  if (value === null || typeof value !== 'object') {
    output.set(prefix, JSON.stringify(value) ?? 'undefined');
    return;
  }
  if (Array.isArray(value)) {
    output.set(prefix, JSON.stringify(value));
    return;
  }
  for (const [key, child] of Object.entries(value as Record<string, unknown>))
    flattenSnapshotValues(child, prefix ? `${prefix}.${key}` : key, output);
};

export function aggregateInspectorSnapshots(
  primary: InspectorEntitySnapshot,
  snapshots: ReadonlyArray<InspectorEntitySnapshot>,
): InspectorEntitySnapshot {
  if (snapshots.length <= 1) return primary;
  const fields = snapshots.map((snapshot) => {
    const values = new Map<string, string>();
    for (const key of ['active', 'tag', 'renderLayerMask', 'mobility', ...aggregateComponentKeys])
      flattenSnapshotValues(snapshot[key as keyof InspectorEntitySnapshot], key, values);
    return values;
  });
  const paths = new Set(fields.flatMap((values) => [...values.keys()]));
  const mixedFields = [...paths]
    .filter((path) => {
      const expected = fields[0].get(path);
      return fields.some((values) => values.get(path) !== expected);
    })
    .sort();
  const presence = aggregateComponentKeys.map((key) => ({
    key,
    count: snapshots.filter((snapshot) => snapshot[key] !== null).length,
  }));
  return {
    ...primary,
    selectionCount: snapshots.length,
    aggregate: {
      mixedFields,
      commonComponents: presence.filter((entry) => entry.count === snapshots.length).map((entry) => entry.key),
      partialComponents: presence
        .filter((entry) => entry.count > 0 && entry.count < snapshots.length)
        .map((entry) => entry.key),
    },
  };
}

export function transformHostPayload(value: InspectorTransform) {
  const rotation = eulerDegreesToQuaternion(value.rotationDegrees);
  return {
    position: [value.position.x, value.position.y, value.position.z],
    rotation: [rotation.x, rotation.y, rotation.z, rotation.w],
    scale: [value.scale.x, value.scale.y, value.scale.z],
  };
}

export function cameraHostPayload(value: InspectorCamera) {
  return {
    ...value,
    clearColor: [value.clearColor.x, value.clearColor.y, value.clearColor.z, value.clearColor.w],
  };
}

export function lightHostPayload(value: InspectorLight) {
  return {
    ...value,
    color: [value.color.x, value.color.y, value.color.z],
  };
}

export const hostEntityKey = (entity: HostEntityId) => `${entity.index}:${entity.generation}`;
