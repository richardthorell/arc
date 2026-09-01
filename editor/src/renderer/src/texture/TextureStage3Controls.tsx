import { useEffect, useRef, useState } from 'react';

import type { AssetItem } from '../services/editorHostTypes';
import { UiPanelSection } from '../ui';
import {
  patchTextureSettings,
  type TextureChannelSource,
  type TextureSettingsPatch,
  type TextureSettingsSnapshot,
} from './textureSettings';
import { useTextureSettings } from './useTextureSettings';

const channelOptions: Array<[TextureChannelSource, string]> = [
  ['red', 'R'],
  ['green', 'G'],
  ['blue', 'B'],
  ['alpha', 'A'],
  ['zero', '0'],
  ['one', '1'],
];

function NumberControl({
  label,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label className="inspector-property texture-inspector-property texture-stage3-number">
      <span className="inspector-property-label">{label}</span>
      <span className="texture-stage3-number-fields">
        <input
          aria-label={`${label} slider`}
          disabled={disabled}
          min={min}
          max={max}
          step={step}
          type="range"
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <input
          aria-label={label}
          className="texture-inspector-input"
          disabled={disabled}
          min={min}
          max={max}
          step={step}
          type="number"
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
      </span>
    </label>
  );
}

export function TextureStage3Controls({ asset }: { asset: AssetItem }) {
  const { settings } = useTextureSettings(asset.guid, asset.generation);
  const [draft, setDraft] = useState<TextureSettingsSnapshot | null>(settings);
  const timer = useRef<number | null>(null);
  useEffect(() => setDraft(settings), [settings]);
  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );
  if (!asset.guid || !draft || asset.readOnly) return null;

  const update = (patch: TextureSettingsPatch) => {
    setDraft((current) => (current ? { ...current, ...patch } : current));
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      void patchTextureSettings(asset.guid!, patch);
    }, 250);
  };
  const normal = draft.semantic === 'normal';
  return (
    <>
      <UiPanelSection className="texture-inspector-section" title="Adjustments">
        {normal && <div className="texture-stage3-note">Color adjustments are bypassed for normal-map semantics.</div>}
        <NumberControl
          disabled={normal}
          label="Brightness"
          min={-4}
          max={4}
          step={0.05}
          value={draft.brightness}
          onChange={(brightness) => update({ brightness })}
        />
        <NumberControl
          disabled={normal}
          label="Gamma"
          min={0.1}
          max={4}
          step={0.05}
          value={draft.gamma}
          onChange={(gamma) => update({ gamma })}
        />
        <NumberControl
          disabled={normal}
          label="Contrast"
          min={0}
          max={2}
          step={0.05}
          value={draft.contrast}
          onChange={(contrast) => update({ contrast })}
        />
        <NumberControl
          disabled={normal}
          label="Saturation"
          min={0}
          max={2}
          step={0.05}
          value={draft.saturation}
          onChange={(saturation) => update({ saturation })}
        />
        <NumberControl
          disabled={normal}
          label="Vibrance"
          min={-1}
          max={1}
          step={0.05}
          value={draft.vibrance}
          onChange={(vibrance) => update({ vibrance })}
        />
        <NumberControl
          disabled={normal}
          label="Tint R"
          min={0}
          max={2}
          step={0.02}
          value={draft.tintR}
          onChange={(tintR) => update({ tintR })}
        />
        <NumberControl
          disabled={normal}
          label="Tint G"
          min={0}
          max={2}
          step={0.02}
          value={draft.tintG}
          onChange={(tintG) => update({ tintG })}
        />
        <NumberControl
          disabled={normal}
          label="Tint B"
          min={0}
          max={2}
          step={0.02}
          value={draft.tintB}
          onChange={(tintB) => update({ tintB })}
        />
      </UiPanelSection>
      <UiPanelSection className="texture-inspector-section" collapsed title="Levels">
        <NumberControl
          disabled={normal}
          label="Input Black"
          min={0}
          max={0.99}
          step={0.01}
          value={draft.inputBlack}
          onChange={(inputBlack) => update({ inputBlack: Math.min(inputBlack, draft.inputWhite - 0.01) })}
        />
        <NumberControl
          disabled={normal}
          label="Input White"
          min={0.01}
          max={1}
          step={0.01}
          value={draft.inputWhite}
          onChange={(inputWhite) => update({ inputWhite: Math.max(inputWhite, draft.inputBlack + 0.01) })}
        />
        <NumberControl
          disabled={normal}
          label="Output Black"
          min={0}
          max={1}
          step={0.01}
          value={draft.outputBlack}
          onChange={(outputBlack) => update({ outputBlack: Math.min(outputBlack, draft.outputWhite) })}
        />
        <NumberControl
          disabled={normal}
          label="Output White"
          min={0}
          max={1}
          step={0.01}
          value={draft.outputWhite}
          onChange={(outputWhite) => update({ outputWhite: Math.max(outputWhite, draft.outputBlack) })}
        />
      </UiPanelSection>
      <UiPanelSection className="texture-inspector-section" collapsed title="Channel Mapping">
        {(['R', 'G', 'B', 'A'] as const).map((channel) => {
          const sourceKey = `channel${channel}` as 'channelR' | 'channelG' | 'channelB' | 'channelA';
          const invertKey = `invert${channel}` as 'invertR' | 'invertG' | 'invertB' | 'invertA';
          return (
            <div className="texture-stage3-channel" key={channel}>
              <span>{channel}</span>
              <select
                aria-label={`${channel} source`}
                value={draft[sourceKey]}
                onChange={(event) => update({ [sourceKey]: event.target.value as TextureChannelSource })}
              >
                {channelOptions.map(([value, label]) => (
                  <option key={value} value={value}>
                    {label}
                  </option>
                ))}
              </select>
              <label>
                <input
                  checked={draft[invertKey]}
                  type="checkbox"
                  onChange={(event) => update({ [invertKey]: event.target.checked })}
                />{' '}
                Invert
              </label>
            </div>
          );
        })}
      </UiPanelSection>
    </>
  );
}
