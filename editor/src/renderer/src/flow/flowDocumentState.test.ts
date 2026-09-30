// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { openEditorDocumentInStore, resetEditorDocuments } from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import {
  disposeFlowDocument,
  flowDocumentPlaySource,
  getFlowDocumentState,
  loadFlowDocument,
  replaceFlowGraph,
  saveFlowDocument,
} from './flowDocumentState';
import { cloneFlowGraph, createFlowAsset } from './flowGraphTypes';

const document: EditorDocument = {
  id: 'flow:flow-guid',
  kind: 'flow',
  title: 'QuickStart.arcflow',
  path: 'Content/QuickStart.arcflow',
  assetGuid: 'flow-guid',
  assetScope: 'project',
  dirty: false,
  readOnly: false,
};

const readText = vi.fn();
const writeText = vi.fn();
const query = vi.fn();
const command = vi.fn();

beforeEach(() => {
  const asset = createFlowAsset('QuickStart');
  readText.mockResolvedValue({ path: document.path, text: `${JSON.stringify(asset)}\n`, modifiedAt: '' });
  writeText.mockResolvedValue({ succeeded: true });
  query.mockResolvedValue({ succeeded: true, payload: { state: 'playing' } });
  command.mockResolvedValue({ succeeded: true });
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: { projects: { readText, writeText }, host: { query, command } },
  });
  openEditorDocumentInStore(document);
});

afterEach(() => {
  disposeFlowDocument(document.id);
  resetEditorDocuments();
  vi.clearAllMocks();
});

describe('Flow document Play sources', () => {
  it('serializes dirty editor-visible state without saving it to disk', async () => {
    expect(await loadFlowDocument(document)).toBe(true);
    const loaded = getFlowDocumentState(document);
    const edited = cloneFlowGraph(loaded.graph);
    edited.viewport = { ...edited.viewport, x: 320, y: -180 };
    replaceFlowGraph(document, edited);

    const source = flowDocumentPlaySource(document);

    expect(source).toMatchObject({ guid: 'flow-guid', revision: loaded.revision + 1 });
    expect(JSON.parse(source?.source ?? '').graph.viewport).toMatchObject({ x: 320, y: -180 });
    expect(writeText).not.toHaveBeenCalled();
  });

  it('publishes the saved generation to an active Play sandbox', async () => {
    expect(await loadFlowDocument(document)).toBe(true);
    const edited = cloneFlowGraph(getFlowDocumentState(document).graph);
    edited.viewport = { ...edited.viewport, zoom: 1.75 };
    replaceFlowGraph(document, edited);

    expect(await saveFlowDocument(document)).toBe(true);

    expect(writeText).toHaveBeenCalledTimes(1);
    expect(command).toHaveBeenCalledWith(
      'runtime.updateFlowSource',
      expect.objectContaining({
        guid: 'flow-guid',
        revision: getFlowDocumentState(document).revision,
        source: expect.stringContaining('"zoom": 1.75'),
      }),
    );
  });
});
