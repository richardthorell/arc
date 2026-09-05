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
  name?: string;
};

export type ViewportMaterialDropResult =
  { succeeded: true; entity: HostEntityId } | { succeeded: false; error: string };

export type ViewportMeshDropIntent = 'create' | 'createChild' | 'replace';

export type ViewportMeshDropResult =
  | { succeeded: true; entity: HostEntityId; intent: ViewportMeshDropIntent }
  | { succeeded: false; error: string };

type WaitOptions = {
  attempts?: number;
  intervalMs?: number;
  sleep?: (milliseconds: number) => Promise<void>;
};

type PickedEntity = {
  entity?: HostEntityId;
  name?: string;
};

const validEntity = (entity: HostEntityId | undefined): entity is HostEntityId =>
  Boolean(entity && entity.index !== 0xffffffff);

const defaultSleep = (milliseconds: number) =>
  new Promise<void>((resolve) => {
    window.setTimeout(resolve, milliseconds);
  });

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

const waitForPickedEntity = async (
  host: HostBridge,
  pickFrameRevision: number | undefined,
  waitOptions: WaitOptions,
): Promise<{ succeeded: true; picked: PickedEntity } | { succeeded: false; error: string }> => {
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
      return { succeeded: true, picked: {} };
    }
    return { succeeded: true, picked: { entity, name: selected.payload?.name } };
  }

  return { succeeded: false, error: lastError || 'Timed out while resolving the viewport drop target' };
};

export const viewportMeshDropIntent = (modifiers: { control?: boolean; shift?: boolean }): ViewportMeshDropIntent =>
  modifiers.shift ? 'replace' : modifiers.control ? 'createChild' : 'create';

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

  const resolved = await waitForPickedEntity(host, pick?.frameRevision, waitOptions);
  if (!resolved.succeeded) return resolved;
  const entity = resolved.picked.entity;
  if (!entity) return { succeeded: false, error: 'No scene object at the drop position' };

  const assignment = (await host.command('entity.setMaterial', {
    entity,
    path: request.path,
  })) as HostResponse;
  if (assignment?.succeeded === false)
    return { succeeded: false, error: assignment.error || 'Could not assign the dropped material' };
  return { succeeded: true, entity };
}

const deleteCreatedEntity = async (host: HostBridge, entity: HostEntityId) => {
  await host.command('entity.delete', { entity }).catch(() => undefined);
};

/**
 * Apply a Content Browser model drop using one unambiguous viewport intent:
 * - drag: create a root entity with the model and the editor's Default Phong material
 * - Ctrl+drag: create the same entity, then parent it to the picked object while preserving world transform
 * - Shift+drag: replace only the picked entity's mesh and preserve its existing material/components
 *
 * The temporary cube creation is intentional: primitive entities are initialized with
 * a MeshRenderer and Default Phong. Assigning the imported mesh clears the procedural
 * primitive metadata while retaining that default material.
 */
export async function instantiateDroppedMeshInViewport(
  host: HostBridge,
  request: {
    viewportId: string;
    x: number;
    y: number;
    path: string;
    control?: boolean;
    shift?: boolean;
  },
  waitOptions: WaitOptions = {},
): Promise<ViewportMeshDropResult> {
  const intent = viewportMeshDropIntent(request);
  const pick = (await host.command('viewport.pick', {
    viewportId: request.viewportId,
    x: request.x,
    y: request.y,
  })) as HostResponse;
  if (pick?.succeeded === false) return { succeeded: false, error: pick.error || 'Viewport placement failed' };

  // viewport.pick resolves selection on a rendered frame. Always wait before
  // creating anything so the delayed pick cannot steal selection from the new entity.
  const resolved = await waitForPickedEntity(host, pick?.frameRevision, waitOptions);
  if (!resolved.succeeded) return resolved;
  const target = resolved.picked.entity;

  if (intent === 'replace') {
    if (!target) return { succeeded: false, error: 'Shift+drop requires a mesh target' };
    const assignment = (await host.command('entity.setMaterial', {
      entity: target,
      path: `__arc_mesh__/${request.path}`,
    })) as HostResponse;
    if (assignment?.succeeded === false)
      return { succeeded: false, error: assignment.error || 'Could not replace the target mesh' };
    return { succeeded: true, entity: target, intent };
  }

  const created = (await host.command('entity.create', { kind: 'cube' })) as HostResponse;
  if (created?.succeeded === false) return { succeeded: false, error: created.error || 'Could not create model entity' };
  const entity = parseEntity(created?.payload);
  if (!entity) return { succeeded: false, error: 'Host did not return the created model entity' };

  const assignment = (await host.command('entity.setMaterial', {
    entity,
    path: `__arc_mesh__/${request.path}`,
  })) as HostResponse;
  if (assignment?.succeeded === false) {
    await deleteCreatedEntity(host, entity);
    return { succeeded: false, error: assignment.error || 'Could not assign the dropped model' };
  }

  const worldPosition = parseWorldPosition(pick?.payload);
  if (worldPosition) {
    const transformed = (await host.command('entity.setTransform', {
      entity,
      transform: { position: worldPosition, rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    })) as HostResponse;
    if (transformed?.succeeded === false) {
      await deleteCreatedEntity(host, entity);
      return { succeeded: false, error: transformed.error || 'Could not place the dropped model' };
    }
  }

  if (intent === 'createChild' && target) {
    const reparented = (await host.command('entity.reparent', {
      entity,
      parent: target,
      preserveWorld: true,
    })) as HostResponse;
    if (reparented?.succeeded === false) {
      await deleteCreatedEntity(host, entity);
      return { succeeded: false, error: reparented.error || 'Could not parent the dropped model' };
    }
  }

  const fileName = request.path.replaceAll('\\', '/').split('/').at(-1) ?? request.path;
  const name = fileName.replace(/\.[^.]+$/, '') || 'Model';
  const renamed = (await host.command('entity.rename', { entity, name })) as HostResponse;
  if (renamed?.succeeded === false) {
    await deleteCreatedEntity(host, entity);
    return { succeeded: false, error: renamed.error || 'Could not name the dropped model' };
  }
  return { succeeded: true, entity, intent };
}
