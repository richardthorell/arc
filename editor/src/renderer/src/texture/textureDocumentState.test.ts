// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  getEditorDocumentsSnapshot,
  openEditorDocumentInStore,
  resetEditorDocuments,
} from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import {
  disposeTextureDocument,
  getTextureDocumentState,
  revertTextureDocument,
  saveTextureDocument,
  stageTextureSettings,
} from './textureDocumentState';
import type { TextureSettingsSnapshot } from './textureSettings';

const document: EditorDocument = {
  id: 'texture:texture-guid',
  kind: 'texture',
  title: 'T_Rock.png',
  path: 'Content/Textures/T_Rock.png',
  assetGuid: 'texture-guid',
  assetScope: 'project',
  dirty: false,
  readOnly: false,
};

const savedSettings = {
  settingsVersion: 8,
  preset: 'color',
  semantic: 'base_color',
  colorSpace: 'srgb',
  brightness: 0,
} as TextureSettingsSnapshot;

const command = vi.fn(async () => ({ succeeded: true }));
const query = vi.fn(async () => ({ succeeded: true, payload: savedSettings }));

beforeEach(() => {
  resetEditorDocuments();
  openEditorDocumentInStore(document);
  command.mockClear();
  query.mockClear();
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: { host: { command, query } },
  });
});

afterEach(() => {
  disposeTextureDocument(document.id);
  resetEditorDocuments();
  Reflect.deleteProperty(window, 'arc');
});

describe('texture document state', () => {
  it('stages edits in memory and saves them through the document Save path', async () => {
    expect(stageTextureSettings(document, { brightness: 0.25, contrast: 1.2 })).toBe(true);
    expect(getTextureDocumentState(document.id).pendingPatch).toEqual({ brightness: 0.25, contrast: 1.2 });
    expect(getEditorDocumentsSnapshot().documents[0]?.dirty).toBe(true);

    expect(await saveTextureDocument(document)).toBe(true);

    expect(command).toHaveBeenCalledWith('texture.settings.patch', {
      guid: 'texture-guid',
      brightness: 0.25,
      contrast: 1.2,
    });
    expect(query).toHaveBeenCalledWith('texture.settings', { guid: 'texture-guid' });
    expect(getTextureDocumentState(document.id).pendingPatch).toEqual({});
    expect(getEditorDocumentsSnapshot().documents[0]?.dirty).toBe(false);
  });

  it('reverts pending edits without writing texture settings', async () => {
    stageTextureSettings(document, { gamma: 1.4 });
    command.mockClear();

    expect(await revertTextureDocument(document)).toBe(true);

    expect(command).not.toHaveBeenCalled();
    expect(query).toHaveBeenCalledWith('texture.settings', { guid: 'texture-guid' });
    expect(getTextureDocumentState(document.id).pendingPatch).toEqual({});
    expect(getEditorDocumentsSnapshot().documents[0]?.dirty).toBe(false);
  });
});
