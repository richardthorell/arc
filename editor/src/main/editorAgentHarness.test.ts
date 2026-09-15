import { describe, expect, it, vi } from 'vitest';

import {
  EditorAgentHarness,
  type AgentAssetWorkspace,
  type AgentHarnessHost,
  type AgentHostResponse,
} from './editorAgentHarness';

const response = (payload: unknown = {}, sceneRevision = 4): AgentHostResponse => ({
  kind: 'response',
  requestId: 1,
  succeeded: true,
  error: '',
  payload,
  sceneRevision,
  worldEpoch: 2,
  frameRevision: 12,
});

class MockHost implements AgentHarnessHost {
  readonly commands: Array<{
    type: string;
    payload: Record<string, unknown>;
    edit?: Record<string, unknown>;
    revision?: number;
  }> = [];
  readonly queries: Array<{ type: string; payload: Record<string, unknown> }> = [];
  captureReady = true;
  frameIndex = 12;
  renderOptions = {
    renderMode: 'shaded',
    visualization: 'standard',
    overlay: 'none',
    shadows: true,
    grid: true,
    environment: {
      sky: true,
      fog: true,
      terrain: true,
      water: true,
      vegetation: true,
      decals: true,
    },
  };

  async command(
    type: string,
    payload: Record<string, unknown> = {},
    edit?: Record<string, unknown>,
    revision?: number,
  ): Promise<AgentHostResponse> {
    this.commands.push({ type, payload, edit, revision });
    if (type === 'viewport.setRenderOptions') this.renderOptions = payload as typeof this.renderOptions;
    if (type === 'entity.create')
      return response({ entity: { index: 9, generation: 1 }, guid: 'created-entity-guid' }, 5);
    if (type === 'entity.setTransform' || type === 'entity.setFlow') return response({}, 6);
    return response({}, type === 'entity.rename' ? 5 : 4);
  }

  async query(type: string, payload: Record<string, unknown> = {}): Promise<AgentHostResponse> {
    this.queries.push({ type, payload });
    if (type === 'gateway.entity') {
      return response({ entity: { index: 7, generation: 3 }, guid: payload.guid, name: 'Rock' });
    }
    if (type === 'viewport.state')
      return response({
        frameIndex: this.frameIndex++,
        width: 2,
        height: 1,
        renderOptions: this.renderOptions,
        camera: { distance: 10 },
      });
    if (type === 'gateway.diagnostics')
      return response({
        renderer: { path: 'deferred', renderScale: 1 },
        shadows: { cascades: 4, localCacheHits: 2, localCacheMisses: 0, fallback: '' },
      });
    if (type === 'viewport.captureResult') {
      return response(
        this.captureReady
          ? {
              captureId: payload.captureId,
              frameIndex: this.frameIndex,
              pending: false,
              camera: {
                position: [0, 2, 4],
                forward: [0, 0, -1],
                up: [0, 1, 0],
                nearPlane: 0.25,
                farPlane: 500,
                renderExtent: [2, 1],
                outputExtent: [2, 1],
              },
              images: [
                {
                  channel: 'color',
                  format: 'rgba8',
                  width: 2,
                  height: 1,
                  data: Buffer.from([255, 0, 0, 255, 0, 128, 0, 255]).toString('base64'),
                },
                {
                  channel: 'objectId',
                  format: 'r32ui',
                  width: 2,
                  height: 1,
                  data: Buffer.from(new Uint32Array([0, 42]).buffer).toString('base64'),
                },
              ],
              objects: [{ id: 42, guid: 'entity-guid' }],
            }
          : { captureId: payload.captureId, pending: true },
      );
    }
    return response({ entities: [], totalEntityCount: 0 });
  }
}

class MemoryAssets implements AgentAssetWorkspace {
  readonly files = new Map<string, string>();

  async exists(path: string): Promise<boolean> {
    return this.files.has(path);
  }

  async create(path: string, contents: string): Promise<void> {
    if (this.files.has(path)) throw new Error(`Asset already exists: ${path}`);
    this.files.set(path, contents);
  }

  async remove(path: string): Promise<void> {
    this.files.delete(path);
  }
}

class FailingCommitHost extends MockHost {
  override async command(
    type: string,
    payload: Record<string, unknown> = {},
    edit?: Record<string, unknown>,
    revision?: number,
  ): Promise<AgentHostResponse> {
    const result = await super.command(type, payload, edit, revision);
    return type === 'history.commitTransaction' ? { ...result, succeeded: false, error: 'commit failed' } : result;
  }
}

describe('EditorAgentHarness', () => {
  it('publishes transport-neutral capabilities for built-in and connected agents', async () => {
    const assets = new MemoryAssets();
    const harness = new EditorAgentHarness(new MockHost(), { assets });
    const capabilities = (await harness.invoke('agent.capabilities', {}, 'reader')) as Record<string, unknown>;

    expect(capabilities).toMatchObject({
      assetAuthoringAvailable: true,
      assetKinds: ['material', 'flow', 'shader'],
    });
    expect(capabilities.editActions).toEqual(
      expect.arrayContaining(['create', 'setTransform', 'setFlow', 'createAsset']),
    );
  });

  it('reports native authority revisions on GUID-first reads', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const result = (await gateway.invoke('scene.getEntity', { guid: 'entity-guid' }, 'reader')) as Record<
      string,
      unknown
    >;
    expect(result).toMatchObject({ guid: 'entity-guid', sceneRevision: 4, worldEpoch: 2, frameRevision: 12 });
    expect(host.queries[0]).toEqual({ type: 'gateway.entity', payload: { guid: 'entity-guid' } });
  });

  it('groups approved edits into one explicit native transaction', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const request = (await gateway.invoke('edit.request', { label: 'Move rock' }, 'writer')) as { id: string };
    expect(gateway.approveEdit(request.id)).toBe(true);
    const session = (await gateway.invoke(
      'edit.begin',
      {
        label: 'Move rock',
        expectedSceneRevision: 4,
      },
      'writer',
    )) as { id: string; expectedSceneRevision: number };
    await gateway.invoke(
      'edit.apply',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        action: 'rename',
        value: { guid: 'entity-guid', name: 'Hero Rock' },
      },
      'writer',
    );
    await gateway.invoke(
      'edit.commit',
      {
        editSessionId: session.id,
        expectedSceneRevision: 5,
      },
      'writer',
    );

    expect(host.commands.map((entry) => entry.type)).toEqual([
      'history.beginTransaction',
      'entity.rename',
      'history.commitTransaction',
    ]);
    expect(host.commands[1].edit).toMatchObject({ phase: 'update', label: 'Move rock' });
    expect(gateway.status().activeEditSession).toBeNull();
    expect(gateway.status().lastCommittedEdit).toMatchObject({
      clientId: 'writer',
      label: 'Move rock',
      sceneRevision: 4,
    });
    await gateway.undoLastCommittedEdit();
    expect(host.commands.at(-1)).toMatchObject({ type: 'history.undo', revision: 4 });
    expect(gateway.status().lastCommittedEdit).toBeNull();
  });

  it('returns a persistent GUID so a newly created entity can be transformed and bound to Flow', async () => {
    const host = new MockHost();
    const harness = new EditorAgentHarness(host);
    const request = harness.requestEdit('writer', 'Create moving actor');
    harness.approveEdit(request.id);
    const session = (await harness.invoke(
      'edit.begin',
      { label: 'Create moving actor', expectedSceneRevision: 4 },
      'writer',
    )) as { id: string };
    const created = (await harness.invoke(
      'edit.apply',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        action: 'create',
        value: { kind: 'cube' },
      },
      'writer',
    )) as { guid: string };
    expect(created.guid).toBe('created-entity-guid');

    await harness.invoke(
      'edit.apply',
      {
        editSessionId: session.id,
        expectedSceneRevision: 5,
        action: 'setTransform',
        value: {
          guid: created.guid,
          transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
        },
      },
      'writer',
    );
    await harness.invoke(
      'edit.apply',
      {
        editSessionId: session.id,
        expectedSceneRevision: 6,
        action: 'setFlow',
        value: { guid: created.guid, path: 'Flows/Patrol.arcflow', enabled: true },
      },
      'writer',
    );

    expect(host.commands.slice(-3).map((command) => command.type)).toEqual([
      'entity.create',
      'entity.setTransform',
      'entity.setFlow',
    ]);
    await harness.invoke('edit.cancel', { editSessionId: session.id }, 'writer');
  });

  it('stages typed assets until commit and drops them on cancel', async () => {
    const assets = new MemoryAssets();
    const harness = new EditorAgentHarness(new MockHost(), { assets });
    const request = harness.requestEdit('writer', 'Author gameplay assets');
    harness.approveEdit(request.id);
    let session = (await harness.invoke(
      'edit.begin',
      { label: 'Author gameplay assets', expectedSceneRevision: 4 },
      'writer',
    )) as { id: string };
    await harness.invoke(
      'edit.apply',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        action: 'createAsset',
        value: { kind: 'material', path: 'Materials/AgentStone.arcmat' },
      },
      'writer',
    );
    expect(assets.files.size).toBe(0);
    await harness.invoke('edit.cancel', { editSessionId: session.id }, 'writer');
    expect(assets.files.size).toBe(0);

    session = (await harness.invoke(
      'edit.begin',
      { label: 'Author gameplay assets', expectedSceneRevision: 4 },
      'writer',
    )) as { id: string };
    for (const value of [
      { kind: 'material', path: 'Materials/AgentStone.arcmat' },
      { kind: 'flow', path: 'Flows/AgentPatrol.arcflow' },
    ]) {
      await harness.invoke(
        'edit.apply',
        {
          editSessionId: session.id,
          expectedSceneRevision: 4,
          action: 'createAsset',
          value,
        },
        'writer',
      );
    }
    expect(harness.status().activeEditSession?.stagedAssetCount).toBe(2);
    await harness.invoke('edit.commit', { editSessionId: session.id, expectedSceneRevision: 4 }, 'writer');

    expect(assets.files.get('Materials/AgentStone.arcmat')).toContain('"version": 4');
    expect(assets.files.get('Flows/AgentPatrol.arcflow')).toContain('"assetType": "flow"');
  });

  it('rolls back newly written assets when the native transaction cannot commit', async () => {
    const assets = new MemoryAssets();
    const harness = new EditorAgentHarness(new FailingCommitHost(), { assets });
    const request = harness.requestEdit('writer', 'Failed asset transaction');
    harness.approveEdit(request.id);
    const session = (await harness.invoke(
      'edit.begin',
      { label: 'Failed asset transaction', expectedSceneRevision: 4 },
      'writer',
    )) as { id: string };
    await harness.invoke(
      'edit.apply',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        action: 'createAsset',
        value: { kind: 'flow', path: 'Flows/Retry.arcflow' },
      },
      'writer',
    );

    await expect(
      harness.invoke('edit.commit', { editSessionId: session.id, expectedSceneRevision: 4 }, 'writer'),
    ).rejects.toThrow(/commit failed/);
    expect(assets.files.size).toBe(0);
    expect(harness.status().activeEditSession?.id).toBe(session.id);
    await harness.invoke('edit.cancel', { editSessionId: session.id }, 'writer');
  });

  it('cancels a live edit and revokes authority on project changes', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const request = gateway.requestEdit('writer', 'Experiment');
    gateway.approveEdit(request.id);
    await gateway.invoke('edit.begin', { label: 'Experiment', expectedSceneRevision: 4 }, 'writer');
    await gateway.invalidateAuthority('Scene opened');
    expect(host.commands.at(-1)?.type).toBe('history.cancelTransaction');
    expect(gateway.status().activeEditSession).toBeNull();
    await expect(
      gateway.invoke(
        'edit.begin',
        {
          label: 'No grant',
          expectedSceneRevision: 4,
        },
        'writer',
      ),
    ).rejects.toThrow(/permission/);
  });

  it('restores live edits when a writer disconnects or its grant expires', async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-01T12:00:00Z'));
    try {
      const host = new MockHost();
      const gateway = new EditorAgentHarness(host);
      let request = gateway.requestEdit('writer', 'Disconnected edit');
      gateway.approveEdit(request.id);
      await gateway.invoke(
        'edit.begin',
        {
          label: 'Disconnected edit',
          expectedSceneRevision: 4,
        },
        'writer',
      );
      await gateway.disconnectClient('writer');
      expect(host.commands.at(-1)?.type).toBe('history.cancelTransaction');
      expect(gateway.status().activeEditSession).toBeNull();

      request = gateway.requestEdit('writer', 'Expired edit');
      gateway.approveEdit(request.id);
      await gateway.invoke(
        'edit.begin',
        {
          label: 'Expired edit',
          expectedSceneRevision: 4,
        },
        'writer',
      );
      vi.advanceTimersByTime(15 * 60 * 1000 + 1);
      await gateway.expireInactiveAuthority();
      expect(host.commands.at(-1)?.type).toBe('history.cancelTransaction');
      expect(gateway.status().activeEditSession).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('queues coherent captures and exposes no scene save method', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const capture = (await gateway.invoke(
      'viewport.observe',
      {
        color: true,
        depth: true,
        objectId: true,
        normals: true,
      },
      'observer',
    )) as Record<string, unknown>;
    expect(capture).toMatchObject({ pending: false, sceneRevision: 4 });
    expect(host.commands.at(-1)?.type).toBe('viewport.capture');
    expect(host.queries.at(-1)?.type).toBe('viewport.captureResult');
    await expect(gateway.invoke('scene.save', {}, 'observer')).rejects.toThrow(/Unsupported agent harness method/);
  });

  it('serializes viewport writers and forwards spatial queries', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    await gateway.invoke('viewport.move', { action: 'look', x: 2, y: 1 }, 'first');
    expect(host.commands.at(-1)).toMatchObject({
      type: 'viewport.cameraInput',
      payload: { lookX: 2, lookY: 1 },
    });
    await expect(gateway.invoke('viewport.move', { action: 'orbit', x: 1, y: 0 }, 'second')).rejects.toThrow(/leased/);
    await gateway.invoke(
      'scene.spatialQuery',
      {
        kind: 'raycast',
        origin: [0, 2, 4],
        direction: [0, -0.2, -1],
      },
      'reader',
    );
    expect(host.queries.at(-1)?.type).toBe('gateway.spatialQuery');
  });

  it('forwards non-persistent viewport diagnostics without requesting scene edit access', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    await gateway.invoke(
      'viewport.setRenderOptions',
      {
        renderMode: 'shaded',
        visualization: 'shadowMask',
        overlay: 'none',
        shadows: false,
        environment: { sky: false, terrain: true },
      },
      'observer',
    );

    expect(host.commands.at(-1)).toMatchObject({
      type: 'viewport.setRenderOptions',
      payload: {
        renderMode: 'shaded',
        visualization: 'shadowMask',
        overlay: 'none',
        shadows: false,
        environment: {
          sky: false,
          fog: true,
          terrain: true,
          water: true,
          vegetation: true,
          decals: true,
        },
      },
    });
  });

  it('atomically applies viewport state, settles, captures, and reports effective evidence', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const result = (await gateway.invoke(
      'viewport.debug',
      {
        renderOptions: { visualization: 'shadowMask', shadows: false },
        waitFrames: 2,
        samplePixels: [{ x: 1, y: 0 }],
      },
      'observer',
    )) as Record<string, unknown>;

    expect(result).toMatchObject({
      operation: 'configure-apply-settle-capture',
      effective: {
        renderOptions: { visualization: 'shadowMask', shadows: false },
      },
      transition: { fromFrame: 12 },
    });
    const capture = result.capture as Record<string, unknown>;
    expect(capture).toMatchObject({
      analysis: { channels: { color: { sampledPixels: 2 } } },
      samples: [
        {
          channels: { objectId: { id: 42, guid: 'entity-guid' } },
        },
      ],
    });
  });

  it('inspects remembered pixels and compares captures without another renderer readback', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const first = (await gateway.invoke('viewport.observe', {}, 'observer')) as { captureId: number };
    const pixel = (await gateway.invoke(
      'viewport.inspectPixel',
      {
        captureId: first.captureId,
        x: 1,
        y: 0,
      },
      'observer',
    )) as Record<string, unknown>;
    expect(pixel).toMatchObject({
      sampledPixel: [1, 0],
      channels: {
        color: { value: [0, 128 / 255, 0, 1] },
        objectId: { id: 42, guid: 'entity-guid' },
      },
    });

    const second = (await gateway.invoke('viewport.observe', {}, 'observer')) as { captureId: number };
    const comparison = (await gateway.invoke(
      'viewport.compare',
      {
        baselineCaptureId: first.captureId,
        currentCaptureId: second.captureId,
      },
      'observer',
    )) as Record<string, unknown>;
    expect(comparison).toMatchObject({
      baselineCaptureId: first.captureId,
      currentCaptureId: second.captureId,
      channels: { color: { comparable: true, meanAbsoluteError: 0 } },
    });
  });

  it('rejects non-project asset paths before they reach the native host', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const request = gateway.requestEdit('writer', 'Unsafe material');
    gateway.approveEdit(request.id);
    const session = (await gateway.invoke(
      'edit.begin',
      {
        label: 'Unsafe material',
        expectedSceneRevision: 4,
      },
      'writer',
    )) as { id: string };
    await expect(
      gateway.invoke(
        'edit.apply',
        {
          editSessionId: session.id,
          expectedSceneRevision: 4,
          action: 'setMaterial',
          value: { guid: 'entity-guid', path: '../outside.arcmat' },
        },
        'writer',
      ),
    ).rejects.toThrow(/content-relative/);
    expect(host.commands.some((entry) => entry.type === 'entity.setMaterial')).toBe(false);
    await gateway.invoke('edit.cancel', { editSessionId: session.id }, 'writer');
  });

  it('waits for sequenced host events and reports current authority revisions', async () => {
    const host = new MockHost();
    const gateway = new EditorAgentHarness(host);
    const waiting = gateway.invoke(
      'events.wait',
      {
        kind: 'selection',
        afterSequence: 0,
        timeoutMs: 1000,
      },
      'observer',
    );
    gateway.recordHostEvent({
      type: 'entity.selected',
      entity: { index: 7, generation: 3 },
      message: 'Rock selected',
      payload: { guid: 'entity-guid' },
    });
    const result = (await waiting) as { event: { type: string; sequence: number } };
    expect(result.event).toMatchObject({ type: 'entity.selected', sequence: 1 });
    expect(gateway.status()).toMatchObject({
      sceneRevision: 4,
      worldEpoch: 2,
      frameRevision: 12,
      eventSequence: 1,
    });
  });
});
