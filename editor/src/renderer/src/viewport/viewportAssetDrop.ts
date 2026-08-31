type HostEntityId = { index: number; generation: number };

type HostResponse<T = unknown> = {
  succeeded?: boolean;
  error?: string;
  payload?: T;
  frameRevision?: number;
};

type HostBridge = {
  command: (type: string, payload?: Record<string, unknown>) => Promise<unknown>;
  query: (type: string, payload?: Record<string, unknown>) => Promise<unknown>;
};

type SelectedEntityPayload = {
  entity?: HostEntityId;
  selectionCount?: number;
};

export type ViewportMaterialDropResult =
  { succeeded: true; entity: HostEntityId } | { succeeded: false; error: string };

type WaitOptions = {
  attempts?: number;
  intervalMs?: number;
  sleep?: (milliseconds: number) => Promise<void>;
};

const validEntity = (entity: HostEntityId | undefined): entity is HostEntityId =>
  Boolean(entity && entity.index !== 0xffffffff);

const defaultSleep = (milliseconds: number) =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });

/**
 * Pick the scene object under a viewport drop and assign the material to that
 * exact entity. Picking resolves on a rendered frame, so wait until the host's
 * frame revision advances before trusting the selected-entity query. This also
 * handles dropping onto an entity that was already selected.
 */
export async function assignDroppedMaterialToViewport(
  host: HostBridge,
  request: { viewportId: string; x: number; y: number; path: string },
  waitOptions: WaitOptions = {},
): Promise<ViewportMaterialDropResult> {
  const pick = (await host.command('viewport.pick', {
    viewportId: request.viewportId,
    x: request.x,
    y: request.y,
  })) as HostResponse;
  if (pick?.succeeded === false) return { succeeded: false, error: pick.error || 'Viewport pick failed' };

  const pickFrameRevision = pick?.frameRevision;
  const attempts = waitOptions.attempts ?? 60;
  const intervalMs = waitOptions.intervalMs ?? 16;
  const sleep = waitOptions.sleep ?? defaultSleep;
  let lastError = '';

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (attempt > 0) await sleep(intervalMs);
    const selected = (await host.query('entity.selected')) as HostResponse<SelectedEntityPayload>;
    if (selected?.succeeded === false) {
      lastError = selected.error || 'Could not read the picked entity';
      continue;
    }

    if (
      typeof pickFrameRevision === 'number' &&
      typeof selected?.frameRevision === 'number' &&
      selected.frameRevision <= pickFrameRevision
    ) {
      continue;
    }

    const entity = selected?.payload?.entity;
    if (!validEntity(entity) || selected.payload?.selectionCount === 0) {
      return { succeeded: false, error: 'No scene object at the drop position' };
    }

    const assignment = (await host.command('entity.setMaterial', {
      entity,
      path: request.path,
    })) as HostResponse;
    if (assignment?.succeeded === false)
      return { succeeded: false, error: assignment.error || 'Could not assign the dropped material' };
    return { succeeded: true, entity };
  }

  return { succeeded: false, error: lastError || 'Timed out while picking the material drop target' };
}

export type ViewportMeshDropResult = { succeeded: true; entity: HostEntityId } | { succeeded: false; error: string };

const parseEntity = (payload: unknown): HostEntityId | undefined => {
  if (!payload || typeof payload !== 'object') return undefined;
  const entity = (payload as { entity?: HostEntityId }).entity;
  return validEntity(entity) ? entity : undefined;
};

const parseWorldPosition = (payload: unknown): [number, number, number] | undefined => {
  if (!payload || typeof payload !== 'object') return undefined;
  const value = (payload as { worldPosition?: unknown }).worldPosition;
  if (!Array.isArray(value) || value.length !== 3 || !value.every((entry) => typeof entry === 'number'))
    return undefined;
  return value as [number, number, number];
};

export async function instantiateDroppedMeshInViewport(
  host: HostBridge,
  request: { viewportId: string; x: number; y: number; path: string },
): Promise<ViewportMeshDropResult> {
  const pick = (await host.command('viewport.pick', {
    viewportId: request.viewportId,
    x: request.x,
    y: request.y,
  })) as HostResponse;
  if (pick?.succeeded === false) return { succeeded: false, error: pick.error || 'Viewport placement failed' };

  const created = (await host.command('entity.create', { kind: 'empty' })) as HostResponse;
  if (created?.succeeded === false) return { succeeded: false, error: created.error || 'Could not create mesh entity' };
  const entity = parseEntity(created?.payload);
  if (!entity) return { succeeded: false, error: 'Host did not return the created mesh entity' };

  const assignment = (await host.command('entity.setMaterial', {
    entity,
    path: `__arc_mesh__/${request.path}`,
  })) as HostResponse;
  if (assignment?.succeeded === false) {
    await host.command('entity.delete', { entity }).catch(() => undefined);
    return { succeeded: false, error: assignment.error || 'Could not assign the dropped mesh' };
  }

  const worldPosition = parseWorldPosition(pick?.payload);
  if (worldPosition) {
    const transformed = (await host.command('entity.setTransform', {
      entity,
      transform: { position: worldPosition, rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    })) as HostResponse;
    if (transformed?.succeeded === false)
      return { succeeded: false, error: transformed.error || 'Could not place the dropped mesh' };
  }

  const fileName = request.path.replaceAll('\\', '/').split('/').at(-1) ?? request.path;
  const name = fileName.replace(/\.[^.]+$/, '') || 'Mesh';
  const renamed = (await host.command('entity.rename', { entity, name })) as HostResponse;
  if (renamed?.succeeded === false)
    return { succeeded: false, error: renamed.error || 'Could not name the dropped mesh' };
  return { succeeded: true, entity };
}
