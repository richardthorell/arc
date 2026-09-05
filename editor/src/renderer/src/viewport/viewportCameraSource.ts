import type { SceneEntity } from '../services/editorHostTypes';

export type SceneCameraSource = {
  id: string;
  name: string;
  position: [number, number, number];
  target: [number, number, number];
};

const degreesToRadians = (value: number) => (value * Math.PI) / 180;

export const cameraTargetFromTransform = (entity: SceneEntity): SceneCameraSource | null => {
  if (entity.kind !== 'camera' || !entity.active || !entity.transform) return null;
  const { position, rotation } = entity.transform;
  const pitch = degreesToRadians(rotation.x);
  const yaw = degreesToRadians(rotation.y);
  const cosPitch = Math.cos(pitch);
  const forward: [number, number, number] = [-Math.sin(yaw) * cosPitch, Math.sin(pitch), -Math.cos(yaw) * cosPitch];
  return {
    id: entity.id,
    name: entity.name,
    position: [position.x, position.y, position.z],
    target: [position.x + forward[0] * 10, position.y + forward[1] * 10, position.z + forward[2] * 10],
  };
};

export const collectSceneCameraSources = (entities: SceneEntity[]): SceneCameraSource[] =>
  entities.flatMap((entity) => {
    const camera = cameraTargetFromTransform(entity);
    return [...(camera ? [camera] : []), ...collectSceneCameraSources(entity.children ?? [])];
  });
