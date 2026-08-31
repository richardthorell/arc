import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiSelect } from '../ui';
import {
  getTextureEditorViewState,
  setTextureEditorViewState,
  updateTextureChannels,
  useTextureEditorViewState,
} from './textureEditorViewState';

import './textureEditorToolbar.css';

const minZoom = 0.25;
const maxZoom = 16;
const clampZoom = (value: number) => Math.min(maxZoom, Math.max(minZoom, value));

export function TextureEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useTextureEditorViewState(document.id);
  const mipLevels = Math.max(1, document.assetSnapshot?.mipLevels ?? 1);
  const maxMip = mipLevels - 1;
  const mipOptions = Array.from({ length: mipLevels }, (_, level) => ({
    value: String(level),
    label: `Mip Level ${level}`,
  }));
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
          <UiButton
            active={state.channels[channel]}
            aria-pressed={state.channels[channel]}
            className={`texture-channel texture-channel-${channel}`}
            key={channel}
            onClick={() => toggleChannel(channel)}
            type="button"
            variant="toolbar"
          >
            {channel.toUpperCase()}
          </UiButton>
        ))}
      </div>

      <span aria-hidden="true" className="texture-toolbar-separator" />

      <UiSelect
        ariaLabel="Mip level"
        className="texture-mip-select"
        disabled={mipLevels <= 1}
        options={mipOptions}
        value={String(state.mipLevel)}
        onValueChange={(value) => setMipLevel(Number(value))}
      />
      <UiButton
        aria-label="Increase mip level"
        className="texture-mip-step"
        disabled={state.mipLevel >= maxMip}
        onClick={() => setMipLevel(state.mipLevel + 1)}
        type="button"
        variant="toolbar"
      >
        +
      </UiButton>
      <UiButton
        aria-label="Decrease mip level"
        className="texture-mip-step"
        disabled={state.mipLevel <= 0}
        onClick={() => setMipLevel(state.mipLevel - 1)}
        type="button"
        variant="toolbar"
      >
        −
      </UiButton>

      <span aria-hidden="true" className="texture-toolbar-separator" />

      <label className="texture-zoom-control">
        <span>Zoom</span>
        <input
          aria-label="Texture zoom"
          max={1600}
          min={25}
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
