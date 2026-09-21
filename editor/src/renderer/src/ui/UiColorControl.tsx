import { useState } from 'react';

import { UiColorPicker, colorToCss } from './UiColorPicker';
import type { UiColorValue } from './UiColorPicker';

import './UiColorControl.css';

export type UiColorControlProps = {
  label: string;
  value: UiColorValue;
  allowAlpha?: boolean;
  mixed?: boolean;
  minChannelValue?: number;
  maxChannelValue?: number;
  onPreview?: (value: UiColorValue) => void;
  onCommit: (value: UiColorValue) => void;
};

export function UiColorControl({
  label,
  value,
  allowAlpha = true,
  mixed = false,
  minChannelValue = 0,
  maxChannelValue = 1,
  onPreview,
  onCommit,
}: UiColorControlProps) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const hdr = maxChannelValue > 1;

  return (
    <div className="ui-color-control">
      <button
        aria-expanded={pickerOpen}
        aria-label={`Open ${label} color picker`}
        className="ui-color-control-swatch"
        onClick={() => setPickerOpen((open) => !open)}
        type="button"
      >
        <span
          aria-hidden="true"
          className={`ui-color-control-swatch-color ${mixed ? 'is-mixed' : ''}`}
          style={mixed ? undefined : { background: colorToCss(value) }}
        />
      </button>
      <div className="ui-color-control-badges" aria-hidden="true">
        <span className="ui-color-control-badge is-channels">{allowAlpha ? 'RGBA' : 'RGB'}</span>
        {hdr && <span className="ui-color-control-badge is-hdr">HDR</span>}
      </div>
      {pickerOpen && (
        <UiColorPicker
          label={label}
          maxChannelValue={maxChannelValue}
          minChannelValue={minChannelValue}
          showAlpha={allowAlpha}
          value={value}
          onClose={() => setPickerOpen(false)}
          onCommit={onCommit}
          onPreview={onPreview ?? (() => undefined)}
        />
      )}
    </div>
  );
}
