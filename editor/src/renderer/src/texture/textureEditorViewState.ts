import { useSyncExternalStore } from 'react';

export type TextureChannels = { r: boolean; g: boolean; b: boolean; a: boolean };
export type TextureEditorViewState = {
  zoom: number;
  mipLevel: number;
  channels: TextureChannels;
  previewMode: 'source' | 'processed' | 'difference';
};

const createInitialState = (): TextureEditorViewState => ({
  zoom: 1,
  mipLevel: 0,
  channels: { r: true, g: true, b: true, a: true },
  previewMode: 'processed',
});

const states = new Map<string, TextureEditorViewState>();
const listeners = new Map<string, Set<() => void>>();

export const getTextureEditorViewState = (id: string) => {
  let state = states.get(id);
  if (!state) {
    state = createInitialState();
    states.set(id, state);
  }
  return state;
};

export const setTextureEditorViewState = (id: string, patch: Partial<TextureEditorViewState>) => {
  const current = getTextureEditorViewState(id);
  states.set(id, { ...current, ...patch });
  listeners.get(id)?.forEach((listener) => listener());
};

export const updateTextureChannels = (id: string, patch: Partial<TextureChannels>) => {
  const current = getTextureEditorViewState(id);
  setTextureEditorViewState(id, { channels: { ...current.channels, ...patch } });
};

export const useTextureEditorViewState = (id: string) =>
  useSyncExternalStore(
    (listener) => {
      const bucket = listeners.get(id) ?? new Set<() => void>();
      bucket.add(listener);
      listeners.set(id, bucket);
      return () => {
        bucket.delete(listener);
        if (!bucket.size) listeners.delete(id);
      };
    },
    () => getTextureEditorViewState(id),
    () => getTextureEditorViewState(id),
  );
