// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import {
  compileMaterialDocument,
  disposeMaterialDocument,
  getMaterialDocumentState,
  loadMaterialDocument,
  replaceMaterialGraph,
  replaceMaterialSettings,
  setMaterialGraphView,
  setMaterialLiveUpdate,
} from './materialDocumentState';
import { cloneMaterialGraph, createDefaultMaterialGraph } from './materialGraphTypes';

const document: EditorDocument = {
  id: 'material-auto-preview-test',
  kind: 'material',
  title: 'Auto Preview',
  path: 'Content/AutoPreview.arcmat',
  assetGuid: '00112233445566778899aabbccddeeff',
  assetScope: 'project',
  dirty: false,
  readOnly: false,
};

const command = vi.fn(async (_type?: string, _payload?: unknown) => ({
  succeeded: true,
  payload: { succeeded: true, diagnostics: [] },
}));
const query = vi.fn(async () => ({ succeeded: false, error: 'thumbnail unavailable' }));

beforeEach(() => {
  vi.useFakeTimers();
  command.mockClear();
  query.mockClear();
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      host: { command, query },
      projects: {
        readText: vi.fn(async () => ({
          text: JSON.stringify({
            version: 4,
            name: 'Auto Preview',
            graph: createDefaultMaterialGraph(),
          }),
        })),
      },
    },
  });
});

afterEach(() => {
  disposeMaterialDocument(document.id);
  vi.useRealTimers();
});

describe('material live preview compilation', () => {
  it('compiles semantic edits but ignores node movement and viewport changes', async () => {
    await loadMaterialDocument(document, true);
    await vi.advanceTimersByTimeAsync(250);
    expect(getMaterialDocumentState(document).compilation.status).toBe('succeeded');
    command.mockClear();

    const initial = getMaterialDocumentState(document);
    const moved = cloneMaterialGraph(initial.graph);
    moved.nodes[0].position = [850, 640];
    moved.viewport = { x: -200, y: 90, zoom: 1.3 };
    replaceMaterialGraph(document, moved, { recordHistory: false });

    await vi.advanceTimersByTimeAsync(250);
    expect(command).not.toHaveBeenCalled();
    expect(getMaterialDocumentState(document).compilation.status).toBe('succeeded');

    const valueEdit = cloneMaterialGraph(getMaterialDocumentState(document).graph);
    const constant = valueEdit.nodes.find((node) => node.type === 'constant');
    expect(constant).toBeDefined();
    constant!.values.value = 0.2;
    replaceMaterialGraph(document, valueEdit);

    await vi.advanceTimersByTimeAsync(250);
    expect(command).toHaveBeenCalledTimes(1);
    expect(command).toHaveBeenCalledWith(
      'shader.compile',
      expect.objectContaining({
        domain: 'materialGraph',
        previewGuid: document.assetGuid,
      }),
    );
  });

  it('keeps a pending semantic compile scheduled while layout continues changing', async () => {
    await loadMaterialDocument(document, true);
    await vi.advanceTimersByTimeAsync(250);
    command.mockClear();

    const semantic = cloneMaterialGraph(getMaterialDocumentState(document).graph);
    semantic.connections.pop();
    replaceMaterialGraph(document, semantic);

    const moved = cloneMaterialGraph(getMaterialDocumentState(document).graph);
    moved.nodes[0].position = [400, 300];
    replaceMaterialGraph(document, moved, { recordHistory: false });

    await vi.advanceTimersByTimeAsync(250);
    expect(command).toHaveBeenCalledTimes(1);
  });
});

describe('material toolbar session controls', () => {
  it('can pause and resume live graph compilation', async () => {
    await loadMaterialDocument(document, true);
    await vi.advanceTimersByTimeAsync(250);
    command.mockClear();

    setMaterialLiveUpdate(document, false);
    const edited = cloneMaterialGraph(getMaterialDocumentState(document).graph);
    const constant = edited.nodes.find((node) => node.type === 'constant');
    expect(constant).toBeDefined();
    constant!.values.value = 0.35;
    replaceMaterialGraph(document, edited);

    await vi.advanceTimersByTimeAsync(250);
    expect(command).not.toHaveBeenCalled();

    setMaterialLiveUpdate(document, true);
    await vi.advanceTimersByTimeAsync(250);
    expect(command).toHaveBeenCalledTimes(1);
  });

  it('keeps graph view preferences in editor session state', async () => {
    await loadMaterialDocument(document, true);

    setMaterialGraphView(document, { showGrid: false, dimUnrelated: true });

    const state = getMaterialDocumentState(document);
    expect(state.showGrid).toBe(false);
    expect(state.dimUnrelated).toBe(true);
  });
});

describe('material settings editing', () => {
  it('keeps authored setting edits in memory until an explicit compile', async () => {
    await loadMaterialDocument(document, true);
    await vi.advanceTimersByTimeAsync(250);
    expect(getMaterialDocumentState(document).compilation.status).toBe('succeeded');
    command.mockClear();

    expect(
      replaceMaterialSettings(document, {
        domain: 'surface',
        blendMode: 'masked',
        shadingModel: 'unlit',
        doubleSided: true,
        castShadows: false,
      }),
    ).toBe(true);

    const updated = getMaterialDocumentState(document);
    expect(updated.asset.blendMode).toBe('masked');
    expect(updated.asset.shadingModel).toBe('unlit');
    expect(updated.asset.doubleSided).toBe(true);
    expect(updated.asset.castShadows).toBe(false);
    expect(updated.compilation.status).toBe('succeeded');

    await vi.advanceTimersByTimeAsync(250);
    expect(command).not.toHaveBeenCalled();

    expect(await compileMaterialDocument(document, { quiet: true })).toBe(true);
    expect(command).toHaveBeenCalledTimes(1);
    const payload = command.mock.calls[0]?.[1] as { previewSource?: string };
    const previewSource = JSON.parse(payload.previewSource ?? '{}');
    expect(previewSource.blendMode).toBe('masked');
    expect(previewSource.shadingModel).toBe('unlit');
    expect(previewSource.doubleSided).toBe(true);
    expect(previewSource.castShadows).toBe(false);
  });

  it('does not compile when a material setting is unchanged', async () => {
    await loadMaterialDocument(document, true);
    await vi.advanceTimersByTimeAsync(250);
    command.mockClear();

    expect(replaceMaterialSettings(document, { blendMode: 'opaque' })).toBe(false);
    await vi.advanceTimersByTimeAsync(250);

    expect(command).not.toHaveBeenCalled();
  });
});
