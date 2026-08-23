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
  | { succeeded: true; entity: HostEntityId }
  | { succeeded: false; error: string };

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
