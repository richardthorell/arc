import type { SceneEntity, Transform as SceneTransform } from '../services/editorHostTypes';
import type { InspectorTransform, Quaternion, Vec3 } from './inspectorTypes';
import { eulerDegreesToQuaternion, quaternionToEulerDegrees } from './inspectorTypes';

const identitySceneTransform = (): SceneTransform => ({
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 },
  scale: { x: 1, y: 1, z: 1 },
});
const multiplyVec = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x * b.x, y: a.y * b.y, z: a.z * b.z });
const divideVec = (a: Vec3, b: Vec3): Vec3 => ({
  x: Math.abs(b.x) > 1e-8 ? a.x / b.x : a.x,
  y: Math.abs(b.y) > 1e-8 ? a.y / b.y : a.y,
  z: Math.abs(b.z) > 1e-8 ? a.z / b.z : a.z,
});
const addVec = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const subtractVec = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const normalizeQuat = (q: Quaternion): Quaternion => {
  const length = Math.hypot(q.x, q.y, q.z, q.w) || 1;
  return { x: q.x / length, y: q.y / length, z: q.z / length, w: q.w / length };
};
const multiplyQuat = (a: Quaternion, b: Quaternion): Quaternion =>
  normalizeQuat({
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  });
const inverseQuat = (q: Quaternion): Quaternion => {
  const normalized = normalizeQuat(q);
  return { x: -normalized.x, y: -normalized.y, z: -normalized.z, w: normalized.w };
};
const rotateVec = (q: Quaternion, value: Vec3): Vec3 => {
  const vector = { x: value.x, y: value.y, z: value.z, w: 0 };
  const result = multiplyQuatRaw(multiplyQuatRaw(q, vector), inverseQuat(q));
  return { x: result.x, y: result.y, z: result.z };
};
const multiplyQuatRaw = (a: Quaternion, b: Quaternion): Quaternion => ({
  x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
  y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
  z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
});

type Trs = { position: Vec3; rotation: Quaternion; scale: Vec3 };
const sceneToTrs = (value: SceneTransform): Trs => ({
  position: value.position,
  rotation: eulerDegreesToQuaternion(value.rotation),
  scale: value.scale,
});
const inspectorToTrs = (value: InspectorTransform): Trs => ({
  position: value.position,
  rotation: eulerDegreesToQuaternion(value.rotationDegrees),
  scale: value.scale,
});
const trsToInspector = (value: Trs): InspectorTransform => ({
  position: value.position,
  rotationDegrees: quaternionToEulerDegrees(value.rotation),
  scale: value.scale,
  rotationQuaternion: normalizeQuat(value.rotation),
});
const compose = (parent: Trs, local: Trs): Trs => ({
  position: addVec(parent.position, rotateVec(parent.rotation, multiplyVec(parent.scale, local.position))),
  rotation: multiplyQuat(parent.rotation, local.rotation),
  scale: multiplyVec(parent.scale, local.scale),
});
const relativeTo = (parent: Trs, world: Trs): Trs => {
  const inverseRotation = inverseQuat(parent.rotation);
  return {
    position: divideVec(rotateVec(inverseRotation, subtractVec(world.position, parent.position)), parent.scale),
    rotation: multiplyQuat(inverseRotation, world.rotation),
    scale: divideVec(world.scale, parent.scale),
  };
};

const flattenScene = (entities: ReadonlyArray<SceneEntity>): SceneEntity[] =>
  entities.flatMap((entity) => [entity, ...flattenScene(entity.children ?? [])]);

function parentWorldTransform(entities: ReadonlyArray<SceneEntity>, entityId: string): Trs | null {
  const flattened = flattenScene(entities);
  const byId = new Map(flattened.map((entity) => [entity.id, entity]));
  const entity = byId.get(entityId);
  if (!entity?.parentId) return null;
  const visited = new Set<string>();
  const worldFor = (id: string): Trs => {
    if (visited.has(id)) return sceneToTrs(identitySceneTransform());
    visited.add(id);
    const current = byId.get(id);
    const local = sceneToTrs(current?.transform ?? identitySceneTransform());
    if (!current?.parentId) return local;
    return compose(worldFor(current.parentId), local);
  };
  return worldFor(entity.parentId);
}

export function inspectorLocalToWorld(
  entities: ReadonlyArray<SceneEntity>,
  entityId: string,
  local: InspectorTransform,
): InspectorTransform {
  const parent = parentWorldTransform(entities, entityId);
  return parent ? trsToInspector(compose(parent, inspectorToTrs(local))) : local;
}

export function inspectorWorldToLocal(
  entities: ReadonlyArray<SceneEntity>,
  entityId: string,
  world: InspectorTransform,
): InspectorTransform {
  const parent = parentWorldTransform(entities, entityId);
  return parent ? trsToInspector(relativeTo(parent, inspectorToTrs(world))) : world;
}
