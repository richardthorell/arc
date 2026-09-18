// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import {
  disposeMaterialDocument,
  getMaterialDocumentState,
  loadMaterialDocument,
  replaceMaterialGraph,
  replaceMaterialSettings,
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

const command = vi.fn(async () => ({
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


describe('material settings editing', () => {
  it('updates authored settings and republishes the live preview without saving first', async () => {
    await loadMaterialDocument(document, true);
    await vi.advanceTimersByTimeAsync(250);
    command.mockClear();

    expect(
      replaceMaterialSettings(document, {
        domain: 'surface',
        blendMode: 'masked',
        shadingModel: 'unlit',
        doubleSided: true,
      }),
    ).toBe(true);

    const updated = getMaterialDocumentState(document);
    expect(updated.asset.blendMode).toBe('masked');
    expect(updated.asset.shadingModel).toBe('unlit');
    expect(updated.asset.doubleSided).toBe(true);
    expect(updated.compilation.status).toBe('idle');

    await vi.advanceTimersByTimeAsync(250);

    expect(command).toHaveBeenCalledTimes(1);
    const payload = command.mock.calls[0]?.[1] as { previewSource?: string };
    const previewSource = JSON.parse(payload.previewSource ?? '{}');
    expect(previewSource.blendMode).toBe('masked');
    expect(previewSource.shadingModel).toBe('unlit');
    expect(previewSource.doubleSided).toBe(true);
    expect(getMaterialDocumentState(document).compilation.status).toBe('succeeded');
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
