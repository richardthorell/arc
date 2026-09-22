import { useSyncExternalStore } from 'react';

import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import {
  getTextureSettings,
  patchTextureSettings,
  type TextureSettingsPatch,
  type TextureSettingsSnapshot,
} from './textureSettings';

export type TextureDocumentState = {
  pendingPatch: TextureSettingsPatch;
  saving: boolean;
  error: string | null;
};

type TextureSettingsSavedDetail = {
  guid: string;
  settings: TextureSettingsSnapshot;
  hadLivePreview: boolean;
};

type TextureSettingsRevertedDetail = {
  guid: string;
  settings: TextureSettingsSnapshot;
};

const states = new Map<string, TextureDocumentState>();
const listeners = new Map<string, Set<() => void>>();
const livePreviewFields = new Set([
  'semantic',
  'colorSpace',
  'brightness',
  'gamma',
  'contrast',
  'saturation',
  'vibrance',
  'tintR',
  'tintG',
  'tintB',
  'inputBlack',
  'inputWhite',
  'outputBlack',
  'outputWhite',
  'channelR',
  'channelG',
  'channelB',
  'channelA',
  'invertR',
  'invertG',
  'invertB',
  'invertA',
  'curvesEnabled',
  'curveMaster',
  'curveR',
  'curveG',
  'curveB',
  'curveA',
]);

const createState = (): TextureDocumentState => ({
  pendingPatch: {},
  saving: false,
  error: null,
});

const ensureState = (documentId: string) => {
  let state = states.get(documentId);
  if (!state) {
    state = createState();
    states.set(documentId, state);
  }
  return state;
};

const publish = (documentId: string, state: TextureDocumentState) => {
  states.set(documentId, state);
  listeners.get(documentId)?.forEach((listener) => listener());
};

const subscribe = (documentId: string, listener: () => void) => {
  const bucket = listeners.get(documentId) ?? new Set<() => void>();
  bucket.add(listener);
  listeners.set(documentId, bucket);
  return () => {
    bucket.delete(listener);
    if (!bucket.size) listeners.delete(documentId);
  };
};

export const hasPendingTextureSettings = (state: TextureDocumentState) => Object.keys(state.pendingPatch).length > 0;

export const hasLiveTexturePreviewEdits = (patch: TextureSettingsPatch) =>
  Object.keys(patch).some((key) => livePreviewFields.has(key));

export const getTextureDocumentState = (documentId: string) => ensureState(documentId);

export const useTextureDocumentState = (documentId: string) =>
  useSyncExternalStore(
    (listener) => subscribe(documentId, listener),
    () => ensureState(documentId),
    () => ensureState(documentId),
  );

export const stageTextureSettings = (document: EditorDocument, patch: TextureSettingsPatch) => {
  if (document.readOnly || !document.assetGuid) return false;
  const current = ensureState(document.id);
  const pendingPatch = { ...current.pendingPatch, ...patch };
  publish(document.id, { ...current, pendingPatch, error: null });
  updateEditorDocumentInStore(document.id, { dirty: true });
  window.dispatchEvent(
    new CustomEvent('arc:texture-settings-preview', {
      detail: { guid: document.assetGuid, patch },
    }),
  );
  return true;
};

export const saveTextureDocument = async (document: EditorDocument): Promise<boolean> => {
  if (document.readOnly || !document.assetGuid) return false;
  const current = ensureState(document.id);
  if (!hasPendingTextureSettings(current)) return true;
  if (current.saving) return false;

  const pendingPatch = current.pendingPatch;
  publish(document.id, { ...current, saving: true, error: null });
  try {
    await patchTextureSettings(document.assetGuid, pendingPatch);
    const settings = await getTextureSettings(document.assetGuid);
    publish(document.id, { pendingPatch: {}, saving: false, error: null });
    updateEditorDocumentInStore(document.id, { dirty: false });
    window.dispatchEvent(
      new CustomEvent<TextureSettingsSavedDetail>('arc:texture-settings-saved', {
        detail: {
          guid: document.assetGuid,
          settings,
          hadLivePreview: hasLiveTexturePreviewEdits(pendingPatch),
        },
      }),
    );
    return true;
  } catch (error) {
    publish(document.id, {
      ...ensureState(document.id),
      saving: false,
      error: error instanceof Error ? error.message : 'Could not save texture settings',
    });
    return false;
  }
};

export const revertTextureDocument = async (document: EditorDocument): Promise<boolean> => {
  if (document.readOnly || !document.assetGuid) return false;
  const current = ensureState(document.id);
  if (!hasPendingTextureSettings(current)) return true;
  if (current.saving) return false;

  publish(document.id, { ...current, saving: true, error: null });
  try {
    const settings = await getTextureSettings(document.assetGuid);
    publish(document.id, { pendingPatch: {}, saving: false, error: null });
    updateEditorDocumentInStore(document.id, { dirty: false });
    window.dispatchEvent(
      new CustomEvent('arc:texture-settings-preview', {
        detail: { guid: document.assetGuid, patch: settings },
      }),
    );
    window.dispatchEvent(
      new CustomEvent<TextureSettingsRevertedDetail>('arc:texture-settings-reverted', {
        detail: { guid: document.assetGuid, settings },
      }),
    );
    return true;
  } catch (error) {
    publish(document.id, {
      ...ensureState(document.id),
      saving: false,
      error: error instanceof Error ? error.message : 'Could not revert texture settings',
    });
    return false;
  }
};

export const disposeTextureDocument = (documentId: string) => {
  states.delete(documentId);
  listeners.delete(documentId);
};
