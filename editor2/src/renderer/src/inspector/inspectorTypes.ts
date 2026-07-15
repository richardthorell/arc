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
};

export type InspectorMeshRenderer = {
  visible: boolean;
  baseColorTint: Vec4;
  hasMaterial: boolean;
  assetBackedMaterial: boolean;
  materialName: string;
  materialPath: string;
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
  meshRenderer: InspectorMeshRenderer | null;
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
  }).nullable(),
  meshRenderer: z.object({
    visible: z.boolean(),
    baseColorTint: vec4Tuple,
    hasMaterial: z.boolean(),
    assetBackedMaterial: z.boolean(),
    materialName: z.string(),
    materialPath: z.string(),
  }).nullable(),
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

export const hostEntityKey = (entity: HostEntityId) => `${entity.index}:${entity.generation}`;
