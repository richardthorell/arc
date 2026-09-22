import { RotateCcw } from 'lucide-react';

import { UiIconButton, UiNumericInput, UiPropertyCard, UiSelect, UiSlider, UiToggleButton } from '../ui';
import type { TextureChannelSource, TextureSettingsPatch, TextureSettingsSnapshot } from './textureSettings';

const channelOptions: Array<{ value: TextureChannelSource; label: string }> = [
  { value: 'red', label: 'R' },
  { value: 'green', label: 'G' },
  { value: 'blue', label: 'B' },
  { value: 'alpha', label: 'A' },
  { value: 'zero', label: '0' },
  { value: 'one', label: '1' },
];

function TextureSliderControl({
  label,
  value,
  min,
  max,
  step,
  disabled,
  defaultValue,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  defaultValue: number;
  onChange: (value: number) => void;
}) {
  const precision = step >= 1 ? 0 : step >= 0.1 ? 1 : 2;
  return (
    <div className="texture-stage3-number-control">
      <UiSlider
        aria-label={`${label} slider`}
        disabled={disabled}
        max={max}
        min={min}
        onValueChange={onChange}
        step={step}
        value={value}
      />
      <UiNumericInput
        ariaLabel={label}
        disabled={disabled}
        max={max}
        min={min}
        onCommit={onChange}
        precision={precision}
        scrubSensitivity={step}
        step={step}
        value={value}
      />
      <UiIconButton
        className="texture-stage3-reset"
        disabled={disabled || Object.is(value, defaultValue)}
        label={`Reset ${label}`}
        onClick={() => onChange(defaultValue)}
        title={`Reset ${label}`}
        type="button"
      >
        <RotateCcw aria-hidden="true" size={12} />
      </UiIconButton>
    </div>
  );
}

export function TextureStage3Controls({
  draft,
  update,
}: {
  draft: TextureSettingsSnapshot;
  update: (patch: TextureSettingsPatch) => void;
}) {

  const normal = draft.semantic === 'normal';
  const adjustmentFields = [
    ...(normal
      ? [
          {
            id: 'normal-note',
            fullWidth: true,
            control: (
              <div className="texture-stage3-note">Color adjustments are bypassed for normal-map semantics.</div>
            ),
          },
        ]
      : []),
    ...[
      ['brightness', 'Brightness', draft.brightness, -4, 4, 0.05, 0],
      ['gamma', 'Gamma', draft.gamma, 0.1, 4, 0.05, 1],
      ['contrast', 'Contrast', draft.contrast, 0, 2, 0.05, 1],
      ['saturation', 'Saturation', draft.saturation, 0, 2, 0.05, 1],
      ['vibrance', 'Vibrance', draft.vibrance, -1, 1, 0.05, 0],
      ['tintR', 'Tint R', draft.tintR, 0, 2, 0.02, 1],
      ['tintG', 'Tint G', draft.tintG, 0, 2, 0.02, 1],
      ['tintB', 'Tint B', draft.tintB, 0, 2, 0.02, 1],
    ].map(([key, label, value, min, max, step, defaultValue]) => ({
      id: String(key),
      label: String(label),
      control: (
        <TextureSliderControl
          defaultValue={Number(defaultValue)}
          disabled={normal}
          label={String(label)}
          max={Number(max)}
          min={Number(min)}
          onChange={(next) => update({ [key as keyof TextureSettingsSnapshot]: next } as TextureSettingsPatch)}
          step={Number(step)}
          value={Number(value)}
        />
      ),
    })),
  ];

  const levelFields = [
    ['inputBlack', 'Input Black', draft.inputBlack, 0, 0.99, 0.01, 0],
    ['inputWhite', 'Input White', draft.inputWhite, 0.01, 1, 0.01, 1],
    ['outputBlack', 'Output Black', draft.outputBlack, 0, 1, 0.01, 0],
    ['outputWhite', 'Output White', draft.outputWhite, 0, 1, 0.01, 1],
  ].map(([key, label, value, min, max, step, defaultValue]) => ({
    id: String(key),
    label: String(label),
    control: (
      <TextureSliderControl
        defaultValue={Number(defaultValue)}
        disabled={normal}
        label={String(label)}
        max={Number(max)}
        min={Number(min)}
        onChange={(next) => {
          if (key === 'inputBlack') update({ inputBlack: Math.min(next, draft.inputWhite - 0.01) });
          else if (key === 'inputWhite') update({ inputWhite: Math.max(next, draft.inputBlack + 0.01) });
          else if (key === 'outputBlack') update({ outputBlack: Math.min(next, draft.outputWhite) });
          else update({ outputWhite: Math.max(next, draft.outputBlack) });
        }}
        step={Number(step)}
        value={Number(value)}
      />
    ),
  }));

  const channelFields = (['R', 'G', 'B', 'A'] as const).map((channel) => {
    const sourceKey = `channel${channel}` as 'channelR' | 'channelG' | 'channelB' | 'channelA';
    const invertKey = `invert${channel}` as 'invertR' | 'invertG' | 'invertB' | 'invertA';
    return {
      id: `channel-${channel}`,
      label: channel,
      control: (
        <div className="texture-stage3-channel-control">
          <UiSelect
            ariaLabel={`${channel} source`}
            onValueChange={(value) => update({ [sourceKey]: value as TextureChannelSource })}
            options={channelOptions}
            value={draft[sourceKey]}
          />
          <UiToggleButton
            aria-label={`Invert ${channel}`}
            checked={draft[invertKey]}
            label="Invert"
            onCheckedChange={(checked) => update({ [invertKey]: checked })}
          />
        </div>
      ),
    };
  });

  return (
    <>
      <UiPropertyCard className="texture-inspector-section" fields={adjustmentFields} title="Adjustments" />
      <UiPropertyCard className="texture-inspector-section" collapsed fields={levelFields} title="Levels" />
      <UiPropertyCard className="texture-inspector-section" collapsed fields={channelFields} title="Channel Mapping" />
    </>
  );
}
