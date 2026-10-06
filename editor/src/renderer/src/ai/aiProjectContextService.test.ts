import { describe, expect, it, vi } from 'vitest';

import type { AiContextCollectionEvent } from '../../../common/aiContextTypes';
import {
  AiProjectContextService,
  createAiAssetContextProvider,
  sanitizeAiContextValue,
  type AiContextProvider,
  type AiProjectContextEnvironment,
  type AiProjectContextHostEvent,
} from './aiProjectContextService';

const projectSnapshot = (guid: string) => ({
  activeProject: {
    descriptor: {
      guid,
      name: `Project ${guid}`,
      engineVersion: '1.0.0',
      defaultScene: { guid: 'scene-guid', expectedType: 'scene', pathHint: 'Content/Main.arcscene' },
      startupScenes: [],
      targetPlatforms: [{ id: 'windows', enabled: true }],
      renderer: { backend: 'vulkan', quality: 'high' },
    },
    compatibility: 'compatible',
    writable: true,
    diagnostics: [],
  },
});

const sectionData = (serviceResult: Awaited<ReturnType<AiProjectContextService['collect']>>, id: string) =>
  serviceResult.sections.find((section) => section.id === id)?.data;

describe('AiProjectContextService', () => {
  it('composes project and native host context with stable entity identities', async () => {
    const query = vi.fn(async (type: string) => {
      if (type === 'scene.hierarchy') {
        return {
          succeeded: true,
          sceneRevision: 12,
          worldEpoch: 3,
          payload: {
            sceneGuid: 'scene-guid',
            entities: [
              {
                entity: { index: 14, generation: 2 },
                guid: 'entity-guid',
                parentGuid: '',
                name: 'Camera',
              },
            ],
          },
        };
      }
      if (type === 'entity.selected') {
        return {
          succeeded: true,
          sceneRevision: 12,
          payload: {
            entity: { index: 14, generation: 2 },
            guid: 'entity-guid',
            selectedGuids: ['entity-guid'],
            components: [{ typeId: 'arc.transform', label: 'Transform' }],
          },
        };
      }
      if (type === 'workspace.documents') {
        return { succeeded: true, payload: { active: 'material-guid', documents: [{ guid: 'material-guid' }] } };
      }
      if (type === 'project.assets') {
        return {
          succeeded: true,
          payload: {
            assets: [
              {
                guid: 'material-guid',
                name: 'Hero Material',
                typeId: 'material',
                path: 'Content/Materials/Hero.arcmat',
                scope: 'project',
                generation: 4,
              },
            ],
          },
        };
      }
      if (type === 'gateway.diagnostics') return { succeeded: true, payload: { diagnostics: [] } };
      if (type === 'viewport.state') {
        return { succeeded: true, frameRevision: 44, payload: { viewportId: 'viewport-1', camera: { fov: 60 } } };
      }
      throw new Error(`Unexpected query ${type}`);
    });
    const environment: AiProjectContextEnvironment = {
      projectSnapshot: async () => projectSnapshot('project-a'),
      hostQuery: query,
      now: () => 1_000,
    };
    const service = new AiProjectContextService(environment);

    const context = await service.collect();

    expect(context.schemaVersion).toBe(1);
    expect(context.projectGuid).toBe('project-a');
    expect(context.revision).toMatchObject({ sceneRevision: 12, worldEpoch: 3, frameRevision: 44 });
    expect(context.sections.map((section) => section.id)).toEqual([
      'project',
      'scene',
      'selection',
      'workspace',
      'assets',
      'diagnostics',
      'viewport',
      'recentChanges',
    ]);
    expect(query.mock.calls.map(([type]) => type)).toEqual([
      'scene.hierarchy',
      'entity.selected',
      'workspace.documents',
      'project.assets',
      'gateway.diagnostics',
      'viewport.state',
    ]);

    const scene = sectionData(context, 'scene') as { entities: Array<Record<string, unknown>> };
    const selection = sectionData(context, 'selection') as Record<string, unknown>;
    expect(scene.entities[0]).toMatchObject({ guid: 'entity-guid', name: 'Camera' });
    expect(scene.entities[0]).not.toHaveProperty('entity');
    expect(selection).toMatchObject({ guid: 'entity-guid', selectedGuids: ['entity-guid'] });
    expect(selection).not.toHaveProperty('entity');
    expect(sectionData(context, 'project')).toMatchObject({
      guid: 'project-a',
      defaultScene: { guid: 'scene-guid', pathHint: 'Content/Main.arcscene' },
    });
    expect(sectionData(context, 'assets')).toMatchObject({
      totalCount: 1,
      returnedCount: 1,
      omittedCount: 0,
      assets: [
        {
          guid: 'material-guid',
          path: 'Content/Materials/Hero.arcmat',
          typeId: 'material',
          scope: 'project',
          generation: 4,
        },
      ],
    });
    expect(context.estimatedCost.approximateTokens).toBeGreaterThan(0);
  });

  it('caches providers until an editor event invalidates the affected context', async () => {
    let now = 2_000;
    let hostEvent: ((event: AiProjectContextHostEvent) => void) | undefined;
    let sceneRevision = 0;
    const collectScene = vi.fn(async () => ({ status: 'ready' as const, data: { revision: ++sceneRevision } }));
    const providers: AiContextProvider[] = [{ id: 'scene', collect: collectScene }];
    const environment: AiProjectContextEnvironment = {
      projectSnapshot: async () => projectSnapshot('project-a'),
      hostQuery: async () => undefined,
      subscribeHostEvents: (listener) => {
        hostEvent = listener;
        return () => {
          hostEvent = undefined;
        };
      },
      now: () => now,
    };
    const service = new AiProjectContextService(environment, { providers, maxAgeMs: 5_000 });
    const observed: AiContextCollectionEvent[] = [];
    service.subscribe((event) => observed.push(event));

    const first = await service.collect();
    now += 100;
    const second = await service.collect();
    expect(collectScene).toHaveBeenCalledTimes(1);
    expect(second.sections[0]?.freshness.cache).toBe('cached');
    expect(second.sections[0]?.freshness.ageMs).toBe(100);

    hostEvent?.({ sequence: 9, type: 'component_changed', message: 'Transform updated' });
    now += 10;
    const third = await service.collect();

    expect(collectScene).toHaveBeenCalledTimes(2);
    expect(third.sections[0]?.freshness.cache).toBe('live');
    expect(third.revision.eventSequence).toBe(9);
    expect(observed).toContainEqual(
      expect.objectContaining({ type: 'context.invalidated', providerIds: expect.arrayContaining(['scene']) }),
    );
    expect(first.collectionId).not.toBe(second.collectionId);
    service.dispose();
  });

  it('bounds asset inventory deterministically and refreshes it after asset changes', async () => {
    let hostEvent: ((event: AiProjectContextHostEvent) => void) | undefined;
    const query = vi.fn(async (type: string) => {
      if (type !== 'project.assets') throw new Error(`Unexpected query ${type}`);
      return {
        succeeded: true,
        payload: {
          assets: [
            { guid: 'z-guid', name: 'Zed', typeId: 'texture', path: 'Content/Z.png', scope: 'project' },
            { guid: 'b-guid', name: 'Beta', typeId: 'material', path: 'Content/B.arcmat', scope: 'project' },
            { guid: 'a-guid', name: 'Alpha', typeId: 'material', path: 'Content/A.arcmat', scope: 'project' },
          ],
        },
      };
    });
    const service = new AiProjectContextService(
      {
        projectSnapshot: async () => projectSnapshot('project-a'),
        hostQuery: query,
        subscribeHostEvents: (listener) => {
          hostEvent = listener;
          return () => {
            hostEvent = undefined;
          };
        },
        now: () => 4_000,
      },
      { providers: [createAiAssetContextProvider(2)], maxAgeMs: 60_000 },
    );

    const first = await service.collect();
    const assets = first.sections[0];
    expect(assets).toMatchObject({ id: 'assets', status: 'ready', truncated: true });
    expect(assets?.data).toEqual({
      assets: [
        {
          guid: 'a-guid',
          name: 'Alpha',
          path: 'Content/A.arcmat',
          scope: 'project',
          state: '',
          typeId: 'material',
        },
        {
          guid: 'b-guid',
          generation: undefined,
          name: 'Beta',
          path: 'Content/B.arcmat',
          scope: 'project',
          state: '',
          typeId: 'material',
        },
      ],
      totalCount: 3,
      returnedCount: 2,
      omittedCount: 1,
      byType: [
        { typeId: 'material', count: 2 },
        { typeId: 'texture', count: 1 },
      ],
    });

    await service.collect();
    expect(query).toHaveBeenCalledTimes(1);
    hostEvent?.({ sequence: 12, type: 'asset_renamed', message: 'Asset renamed' });
    await service.collect();
    expect(query).toHaveBeenCalledTimes(2);
    service.dispose();
  });

  it('never reuses cached context after the active project changes', async () => {
    let guid = 'project-a';
    const collectScene = vi.fn(async ({ projectGuid }: { projectGuid: string | null }) => ({
      status: 'ready' as const,
      data: { projectGuid },
    }));
    const provider: AiContextProvider = { id: 'scene', collect: collectScene };
    const environment: AiProjectContextEnvironment = {
      projectSnapshot: async () => projectSnapshot(guid),
      hostQuery: async () => undefined,
      now: () => 5_000,
    };
    const service = new AiProjectContextService(environment, { providers: [provider], maxAgeMs: 60_000 });

    expect(sectionData(await service.collect(), 'scene')).toEqual({ projectGuid: 'project-a' });
    guid = 'project-b';
    expect(sectionData(await service.collect(), 'scene')).toEqual({ projectGuid: 'project-b' });
    expect(collectScene).toHaveBeenCalledTimes(2);
  });

  it('isolates provider failures and applies deterministic context budgets', async () => {
    const providers: AiContextProvider[] = [
      {
        id: 'scene',
        collect: async () => {
          throw new Error('scene unavailable');
        },
      },
      {
        id: 'diagnostics',
        collect: async () => ({
          data: {
            message: 'abcdefghijklmnopqrstuvwxyz',
            values: [1, 2, 3, 4],
            nested: { z: 3, a: 1, b: 2 },
          },
        }),
      },
    ];
    const service = new AiProjectContextService(
      {
        projectSnapshot: async () => projectSnapshot('project-a'),
        hostQuery: async () => undefined,
        now: () => 8_000,
      },
      {
        providers,
        limits: { maxStringLength: 8, maxArrayItems: 2, maxObjectKeys: 2 },
      },
    );

    const context = await service.collect();
    const failed = context.sections.find((section) => section.id === 'scene');
    const diagnostics = context.sections.find((section) => section.id === 'diagnostics');

    expect(failed).toMatchObject({ status: 'error', error: 'scene unavailable' });
    expect(diagnostics?.status).toBe('ready');
    expect(diagnostics?.truncated).toBe(true);
    expect(diagnostics?.estimatedCost.characters).toBeGreaterThan(0);
  });

  it('bounds deep and circular values without mutating source data', () => {
    const source: { label: string; child?: unknown } = { label: 'root' };
    source.child = source;

    const sanitized = sanitizeAiContextValue(source, { maxDepth: 4 });

    expect(sanitized.truncated).toBe(true);
    expect(sanitized.value).toEqual({ child: '[Circular]', label: 'root' });
    expect(source.child).toBe(source);
  });
});
