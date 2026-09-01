export type ViewportFlyMovement = {
  moveRight: number;
  moveUp: number;
  moveForward: number;
};

export const viewportFlyMovementCodes = new Set(['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE']);

export function viewportFlyMovement(keys: ReadonlySet<string>, distance: number): ViewportFlyMovement | null {
  const right = Number(keys.has('KeyD')) - Number(keys.has('KeyA'));
  const up = Number(keys.has('KeyE')) - Number(keys.has('KeyQ'));
  const forward = Number(keys.has('KeyW')) - Number(keys.has('KeyS'));
  const length = Math.hypot(right, up, forward);
  if (length === 0 || !Number.isFinite(distance) || distance <= 0) return null;
  const scale = distance / length;
  return { moveRight: right * scale, moveUp: up * scale, moveForward: forward * scale };
}
