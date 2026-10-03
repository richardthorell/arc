// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';

import type { AiContextSection, AiProjectContextSnapshot } from '../../../common/aiContextTypes';
import {
  aiContextReferenceFromCandidate,
  captureAiViewportReference,
  collectAiContextPickerCandidates,
  createAiAssetContextProvider,
} from './aiContextPicker';

const readySection = (id: AiContextSection['id'], data: AiContextSection['data']): AiContextSection => ({
  id,
  status: 'ready',
  data,
  truncated: false,
  freshness: {
    capturedAt: '2026-10-03T00:00:00.000Z',
    ageMs: 0,
    cache: 'live',
    revision: { sceneRevision: 12, eventSequence: 40 },
  },
  estimatedCost: { characters: 10, approximateTokens: 3 },
});

const snapshot = (): AiProjectContextSnapshot => ({
  schemaVersion: 1,
  collectionId: 'collection-1',
  projectGuid: 'project-guid',
  capturedAt: '2026-10-03T00:00:00.000Z',
  revision: { sceneRevision: 12, worldEpoch: 3, frameRevision: 44, eventSequence: 40 },
  sections: [
    readySection('selection', { guid: 'entity-camera', selectedGuids: ['entity-camera'] }),
    readySection('scene', {
      sceneGuid: 'scene-main',
      entities: [
        { guid: 'entity-camera', name: 'Camera' },
        { guid: 'entity-light', name: 'Key Light' },
      ],
    }),
    readySection('workspace', { active: 'material-guid', documents: [{ guid: 'material-guid' }] }),
    readySection('assets', {
      assets: [
        {
          guid: 'asset-rock',
          path: 'Content/Props/HeroRock.glb',
          typeId: 'arc.mesh',
          state: 'ready',
          generation: 7,
        },
      ],
    }),
    readySection('diagnostics', { warnings: 1 }),
    readySection('viewport', { viewportId: 'viewport-1', camera: { fov: 60 } }),
  ],
  estimatedCost: { characters: 100, approximateTokens: 25 },
});

describe('AI context picker model', () => {
  it('builds quick, entity, and asset candidates from the shared context snapshot', () => {
    const candidates = collectAiContextPickerCandidates(snapshot());

    expect(candidates.map((candidate) => candidate.id)).toEqual(
      expect.arrayContaining([
        'selection:current',
        'scene:scene-main',
        'workspace:material-guid',
        'viewport:viewport-1',
        'diagnostics:current',
        'entity:entity-camera',
        'entity:entity-light',
        'asset:asset-rock',
      ]),
    );
  });

  it('stamps references with project and freshness identity', () => {
    const current = snapshot();
    const asset = collectAiContextPickerCandidates(current).find((candidate) => candidate.id === 'asset:asset-rock');
    expect(asset).toBeDefined();

    const reference = aiContextReferenceFromCandidate(current, asset!);
    expect(reference).toMatchObject({
      id: 'asset:asset-rock',
      kind: 'asset',
      stableId: 'asset-rock',
      metadata: {
        projectGuid: 'project-guid',
        section: 'assets',
        assetGeneration: 7,
        revision: {
          sceneRevision: 12,
          worldEpoch: 3,
          frameRevision: 44,
          eventSequence: 40,
        },
      },
    });
  });

  it('captures the streamed viewport as a frozen image attachment', () => {
    const canvas = document.createElement('canvas');
    canvas.id = 'arc-viewport-surface-viewport-1';
    canvas.width = 640;
    canvas.height = 360;
    vi.spyOn(canvas, 'toDataURL').mockReturnValue('data:image/png;base64,capture');
    document.body.append(canvas);

    const reference = captureAiViewportReference(snapshot());

    expect(reference.kind).toBe('viewportCapture');
    expect(reference.metadata).toMatchObject({ projectGuid: 'project-guid', viewportId: 'viewport-1', width: 640, height: 360 });
    expect(reference.metadata).not.toHaveProperty('revision');
    expect(reference.attachment).toEqual({
      type: 'image',
      mimeType: 'image/png',
      uri: 'data:image/png;base64,capture',
      alt: 'ARC viewport-1 capture',
    });
    canvas.remove();
  });

  it('uses the host asset registry as an explicit-only context provider', async () => {
    const hostQuery = vi.fn().mockResolvedValue({
      succeeded: true,
      payload: {
        assets: [{ guid: 'asset-guid', path: 'Content/Test.arcscene', typeId: 'arc.scene', state: 'ready', generation: 4 }],
      },
    });
    const provider = createAiAssetContextProvider();

    const result = await provider.collect({
      environment: { projectSnapshot: vi.fn(), hostQuery },
      projectGuid: 'project-guid',
      projectSnapshot: {},
      recentChanges: [],
    });

    expect(hostQuery).toHaveBeenCalledWith('project.assets', {});
    expect(result).toMatchObject({
      status: 'ready',
      data: { assets: [{ guid: 'asset-guid', path: 'Content/Test.arcscene', generation: 4 }] },
    });
  });
});
