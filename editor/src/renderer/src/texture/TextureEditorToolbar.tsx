import type { EditorDocument } from '../editors/editorTypes';
import { UiIconButton } from '../ui';
import {
  getTextureEditorViewState,
  setTextureEditorViewState,
  updateTextureChannels,
  useTextureEditorViewState,
} from './textureEditorViewState';

import './textureEditorToolbar.css';

const minZoom = 0.05;
const maxZoom = 16;
const clampZoom = (value: number) => Math.min(maxZoom, Math.max(minZoom, value));

export function TextureEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useTextureEditorViewState(document.id);
  const mipLevels = Math.max(1, document.assetSnapshot?.mipLevels ?? 1);
  const maxMip = mipLevels - 1;
  const setMipLevel = (value: number) =>
    setTextureEditorViewState(document.id, { mipLevel: Math.max(0, Math.min(maxMip, value)) });
  const toggleChannel = (channel: keyof typeof state.channels) =>
    updateTextureChannels(document.id, {
      [channel]: !getTextureEditorViewState(document.id).channels[channel],
    });

  return (
    <div aria-label="Texture editor toolbar" className="texture-editor-toolbar">
      <div aria-label="Texture channels" className="texture-channel-group">
        {(['r', 'g', 'b', 'a'] as const).map((channel) => (
          <button
            aria-pressed={state.channels[channel]}
            className={`texture-channel texture-channel-${channel} ${state.channels[channel] ? 'is-active' : ''}`}
            key={channel}
            onClick={() => toggleChannel(channel)}
            type="button"
          >
            {channel.toUpperCase()}
          </button>
        ))}
      </div>

      <span aria-hidden="true" className="texture-toolbar-separator" />

      <select
        aria-label="Mip level"
        disabled={mipLevels <= 1}
        onChange={(event) => setMipLevel(Number(event.target.value))}
        value={state.mipLevel}
      >
        {Array.from({ length: mipLevels }, (_, level) => (
          <option key={level} value={level}>
            Mip Level {level}
          </option>
        ))}
      </select>
      <UiIconButton
        disabled={state.mipLevel >= maxMip}
        label="Increase mip level"
        onClick={() => setMipLevel(state.mipLevel + 1)}
        type="button"
      >
        +
      </UiIconButton>
      <UiIconButton
        disabled={state.mipLevel <= 0}
        label="Decrease mip level"
        onClick={() => setMipLevel(state.mipLevel - 1)}
        type="button"
      >
        −
      </UiIconButton>

      <span aria-hidden="true" className="texture-toolbar-separator" />

      <label className="texture-zoom-control">
        <span>Zoom</span>
        <input
          aria-label="Texture zoom"
          max={1600}
          min={5}
          onChange={(event) =>
            setTextureEditorViewState(document.id, {
              zoom: clampZoom(Number(event.target.value) / 100),
            })
          }
          step={5}
          type="range"
          value={Math.round(state.zoom * 100)}
        />
        <output>{Math.round(state.zoom * 100)}%</output>
      </label>
    </div>
  );
}
