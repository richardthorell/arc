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
  useColorTemperature: boolean;
  temperatureKelvin: number;
};

export type InspectorMeshRenderer = {
  visible: boolean;
  baseColorTint: Vec4;
  hasMaterial: boolean;
  assetBackedMaterial: boolean;
  materialName: string;
  materialPath: string;
};

export type InspectorTerrain = {
  enabled: boolean;
  size: number;
  resolution: number;
  chunkQuads: number;
  receiveShadows: boolean;
  contentRevision: number;
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
};

export type HostComponentSnapshot = {
  kind: string;
  label: string;
  editable: boolean;
};

export type InspectorEntitySnapshot = {
  entity: HostEntityId;
  name: string;
  tag: string;
  active: boolean;
  renderLayerMask: number;
  transform: InspectorTransform | null;
  camera: InspectorCamera | null;
  light: InspectorLight | null;
  meshRenderer: InspectorMeshRenderer | null;
  terrain: InspectorTerrain | null;
  prefab: InspectorPrefab | null;
  components: HostComponentSnapshot[];
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
  name: z.string(),
  tag: z.string(),
  active: z.boolean(),
  renderLayerMask: z.number().int().nonnegative(),
  transform: z.object({
    position: vec3Tuple,
    rotation: quaternionTuple,
    scale: vec3Tuple,
  }).nullable(),
  camera: z.object({
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
  }).nullable(),
  light: z.object({
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
    useColorTemperature: z.boolean(),
    temperatureKelvin: finiteNumber.min(1000).max(40000),
  }).nullable().default(null),
  meshRenderer: z.object({
    visible: z.boolean(),
    baseColorTint: vec4Tuple,
    hasMaterial: z.boolean(),
    assetBackedMaterial: z.boolean(),
    materialName: z.string(),
    materialPath: z.string(),
  }).nullable(),
  terrain: z.object({
    enabled: z.boolean(),
    size: finiteNumber.positive(),
    resolution: z.number().int().min(3),
    chunkQuads: z.number().int().positive(),
    receiveShadows: z.boolean(),
    contentRevision: z.number().int().nonnegative(),
    brushTool: z.enum(['sculpt', 'smooth', 'flatten', 'paint']),
    brushRadius: finiteNumber.min(0.25).max(128),
    brushStrength: finiteNumber.positive().max(1),
    brushFalloff: finiteNumber.min(0).max(1),
    activeLayer: z.number().int().min(0).max(3),
    layers: z.array(z.object({ name: z.string(), baseColorPath: z.string() })).length(4),
  }).nullable().default(null),
  prefab: z.object({
    prefabGuid: z.string(),
    prefabPath: z.string(),
    overrideCount: z.number().int().nonnegative(),
    sourceMissing: z.boolean(),
  }).nullable().default(null),
  components: z.array(z.object({
    kind: z.string(),
    label: z.string(),
    editable: z.boolean(),
  })),
});

const tupleToVec3 = (value: [number, number, number]): Vec3 => ({ x: value[0], y: value[1], z: value[2] });
const tupleToVec4 = (value: [number, number, number, number]): Vec4 => ({ x: value[0], y: value[1], z: value[2], w: value[3] });
const tupleToQuaternion = (value: [number, number, number, number]): Quaternion => ({ x: value[0], y: value[1], z: value[2], w: value[3] });
const radiansToDegrees = (value: number) => value * 180 / Math.PI;
const degreesToRadians = (value: number) => value * Math.PI / 180;

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
    transform: parsed.transform && rotationQuaternion ? {
      position: tupleToVec3(parsed.transform.position),
      rotationDegrees: quaternionToEulerDegrees(rotationQuaternion),
      scale: tupleToVec3(parsed.transform.scale),
      rotationQuaternion,
    } : null,
    camera: parsed.camera ? {
      ...parsed.camera,
      clearColor: tupleToVec4(parsed.camera.clearColor),
    } : null,
    light: parsed.light ? {
      ...parsed.light,
      color: { ...tupleToVec3(parsed.light.color), w: 1 },
    } : null,
    meshRenderer: parsed.meshRenderer ? {
      ...parsed.meshRenderer,
      baseColorTint: tupleToVec4(parsed.meshRenderer.baseColorTint),
    } : null,
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
