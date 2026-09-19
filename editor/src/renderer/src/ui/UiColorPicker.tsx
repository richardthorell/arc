import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, PointerEvent as ReactPointerEvent } from 'react';
import { Copy, Pipette } from 'lucide-react';
import { createPortal } from 'react-dom';

import { UiButton } from './UiButton';
import { UiDialog } from './UiDialog';
import { UiDropdown } from './UiDropdown';
import { UiIconButton } from './UiIconButton';

import './UiColorPicker.css';

export type UiColorValue = { x: number; y: number; z: number; w: number };

type HsvColor = { h: number; s: number; v: number };
type ColorMode = 'rgb' | 'hsv' | 'linear';

type EyeDropperResult = { sRGBHex: string };
type EyeDropperInstance = { open: () => Promise<EyeDropperResult> };
type EyeDropperConstructor = new () => EyeDropperInstance;

export type UiColorPickerProps = {
  label: string;
  showAlpha?: boolean;
  minChannelValue?: number;
  maxChannelValue?: number;
  value: UiColorValue;
  onClose: () => void;
  onCommit: (value: UiColorValue) => void;
  onPreview: (value: UiColorValue) => void;
};

const pickerWidth = 860;
const pickerEstimatedHeight = 560;
const pickerViewportMargin = 8;
const colorModeOptions = [
  { value: 'rgb' as const, label: 'RGB' },
  { value: 'hsv' as const, label: 'HSV' },
  { value: 'linear' as const, label: 'Linear RGB' },
];
const colorPresetOptions = [{ value: 'custom' as const, label: 'Custom' }];

const centeredPickerPosition = () => ({
  x: Math.max(pickerViewportMargin, (window.innerWidth - pickerWidth) / 2),
  y: Math.max(pickerViewportMargin, (window.innerHeight - pickerEstimatedHeight) / 2),
});

const clamp = (value: number, min = 0, max = 1) => Math.min(Math.max(value, min), max);
const wrapHue = (hue: number) => ((hue % 360) + 360) % 360;

export const linearToSrgb = (value: number) => {
  const channel = clamp(value);
  return channel <= 0.0031308 ? channel * 12.92 : 1.055 * Math.pow(channel, 1 / 2.4) - 0.055;
};

export const srgbToLinear = (value: number) => {
  const channel = clamp(value);
  return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
};

const byteHex = (value: number) =>
  Math.round(clamp(value) * 255)
    .toString(16)
    .padStart(2, '0')
    .toUpperCase();

const displayScale = (value: UiColorValue) => Math.max(1, value.x, value.y, value.z);
const normalizeForDisplay = (value: UiColorValue): UiColorValue => {
  const scale = displayScale(value);
  return { x: value.x / scale, y: value.y / scale, z: value.z / scale, w: value.w };
};

export const colorToHex = (value: UiColorValue, includeAlpha = true) => {
  const display = normalizeForDisplay(value);
  const rgb = `${byteHex(linearToSrgb(display.x))}${byteHex(linearToSrgb(display.y))}${byteHex(linearToSrgb(display.z))}`;
  return `#${rgb}${includeAlpha ? byteHex(value.w) : ''}`;
};

export const hexToLinearColor = (hex: string, fallbackAlpha = 1): UiColorValue | null => {
  const token = hex.trim().replace(/^#/, '');
  if (!/^[\dA-Fa-f]{6}([\dA-Fa-f]{2})?$/.test(token)) return null;
  const channel = (offset: number) => Number.parseInt(token.slice(offset, offset + 2), 16) / 255;
  return {
    x: srgbToLinear(channel(0)),
    y: srgbToLinear(channel(2)),
    z: srgbToLinear(channel(4)),
    w: token.length === 8 ? channel(6) : fallbackAlpha,
  };
};

export const linearColorToHsv = (value: UiColorValue): HsvColor => {
  const display = normalizeForDisplay(value);
  const r = linearToSrgb(display.x);
  const g = linearToSrgb(display.y);
  const b = linearToSrgb(display.z);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  let h = 0;
  if (delta > 1e-7) {
    if (max === r) h = 60 * (((g - b) / delta) % 6);
    else if (max === g) h = 60 * ((b - r) / delta + 2);
    else h = 60 * ((r - g) / delta + 4);
  }
  return { h: wrapHue(h), s: max <= 1e-7 ? 0 : delta / max, v: max };
};

export const hsvToLinearColor = (hsv: HsvColor, alpha: number): UiColorValue => {
  const hue = wrapHue(hsv.h);
  const saturation = clamp(hsv.s);
  const value = clamp(hsv.v);
  const chroma = value * saturation;
  const x = chroma * (1 - Math.abs(((hue / 60) % 2) - 1));
  const match = value - chroma;
  let rgb: [number, number, number];
  if (hue < 60) rgb = [chroma, x, 0];
  else if (hue < 120) rgb = [x, chroma, 0];
  else if (hue < 180) rgb = [0, chroma, x];
  else if (hue < 240) rgb = [0, x, chroma];
  else if (hue < 300) rgb = [x, 0, chroma];
  else rgb = [chroma, 0, x];
  return {
    x: srgbToLinear(rgb[0] + match),
    y: srgbToLinear(rgb[1] + match),
    z: srgbToLinear(rgb[2] + match),
    w: clamp(alpha),
  };
};

export const colorToCss = (value: UiColorValue) => {
  const display = normalizeForDisplay(value);
  return `rgba(${Math.round(linearToSrgb(display.x) * 255)}, ${Math.round(linearToSrgb(display.y) * 255)}, ${Math.round(linearToSrgb(display.z) * 255)}, ${clamp(value.w)})`;
};

export function UiColorPicker({
  label,
  value,
  showAlpha = true,
  minChannelValue = 0,
  maxChannelValue = 1,
  onClose,
  onCommit,
  onPreview,
}: UiColorPickerProps) {
  const [draft, setDraft] = useState(value);
  const [mode, setMode] = useState<ColorMode>('rgb');
  const original = useRef(value);
  const latest = useRef(value);
  const previewFrame = useRef<number | null>(null);
  const pendingPreview = useRef(value);
  const hsv = linearColorToHsv(draft);
  const hdr = maxChannelValue > 1;
  const initialPosition = centeredPickerPosition();
  const hdrScale = displayScale(draft);
  const applyHdrScale = (color: UiColorValue) =>
    hdr ? { ...color, x: color.x * hdrScale, y: color.y * hdrScale, z: color.z * hdrScale } : color;
  const eyeDropper = (window as unknown as { EyeDropper?: EyeDropperConstructor }).EyeDropper;

  useEffect(() => {
    setDraft(value);
    latest.current = value;
  }, [value]);

  useEffect(() => {
    const closeFromKeyboard = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      onCommit(original.current);
      onClose();
    };
    document.addEventListener('keydown', closeFromKeyboard);
    return () => {
      document.removeEventListener('keydown', closeFromKeyboard);
      if (previewFrame.current !== null) window.cancelAnimationFrame(previewFrame.current);
    };
  }, [onClose, onCommit]);

  const emit = (next: UiColorValue, final: boolean) => {
    latest.current = next;
    pendingPreview.current = next;
    setDraft(next);

    if (previewFrame.current !== null) {
      window.cancelAnimationFrame(previewFrame.current);
      previewFrame.current = null;
    }

    if (final) {
      onPreview(next);
      return;
    }

    previewFrame.current = window.requestAnimationFrame(() => {
      previewFrame.current = null;
      onPreview(pendingPreview.current);
    });
  };

  const updateHue = (event: ReactPointerEvent<HTMLDivElement>, final: boolean) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - (bounds.left + bounds.width / 2);
    const y = event.clientY - (bounds.top + bounds.height / 2);
    const hue = wrapHue((Math.atan2(y, x) * 180) / Math.PI);
    emit(applyHdrScale(hsvToLinearColor({ ...hsv, h: hue }, draft.w)), final);
  };

  const huePointer = (event: ReactPointerEvent<HTMLDivElement>, final: boolean) => {
    if (event.type === 'pointerdown') event.currentTarget.setPointerCapture(event.pointerId);
    if (event.type === 'pointermove' && !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    updateHue(event, final);
    if (final && event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const updateSpectrum = (event: ReactPointerEvent<HTMLDivElement>, final: boolean) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const saturation = clamp((event.clientX - bounds.left) / Math.max(bounds.width, 1));
    const brightness = 1 - clamp((event.clientY - bounds.top) / Math.max(bounds.height, 1));
    emit(applyHdrScale(hsvToLinearColor({ h: hsv.h, s: saturation, v: brightness }, draft.w)), final);
  };

  const spectrumPointer = (event: ReactPointerEvent<HTMLDivElement>, final: boolean) => {
    event.stopPropagation();
    if (event.type === 'pointerdown') event.currentTarget.setPointerCapture(event.pointerId);
    if (event.type === 'pointermove' && !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    updateSpectrum(event, final);
    if (final && event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const currentCss = colorToCss(draft);
  const originalCss = colorToCss(original.current);
  const hueCss = colorToCss(hsvToLinearColor({ h: hsv.h, s: 1, v: 1 }, 1));
  const swatchStyle = (color: string) => ({ '--arc-picker-color': color }) as CSSProperties;
  const hueRadians = (hsv.h * Math.PI) / 180;
  const hueCursorStyle = {
    left: `${50 + Math.cos(hueRadians) * 44}%`,
    top: `${50 + Math.sin(hueRadians) * 44}%`,
  };

  const finish = (next: UiColorValue) => {
    if (previewFrame.current !== null) {
      window.cancelAnimationFrame(previewFrame.current);
      previewFrame.current = null;
    }
    onCommit(next);
    onClose();
  };
  const cancel = () => finish(original.current);
  const accept = () => finish(latest.current);

  const setRgbChannel = (index: number, next: number) => {
    if (index === 3) {
      emit({ ...draft, w: clamp(next) }, true);
      return;
    }

    if (mode === 'linear') {
      const channels = [draft.x, draft.y, draft.z];
      channels[index] = next;
      emit({ x: channels[0], y: channels[1], z: channels[2], w: draft.w }, true);
      return;
    }

    const rgb = [
      linearToSrgb(draft.x / hdrScale) * 255,
      linearToSrgb(draft.y / hdrScale) * 255,
      linearToSrgb(draft.z / hdrScale) * 255,
    ];
    rgb[index] = next;
    emit(
      {
        x: srgbToLinear(rgb[0] / 255) * (hdr ? hdrScale : 1),
        y: srgbToLinear(rgb[1] / 255) * (hdr ? hdrScale : 1),
        z: srgbToLinear(rgb[2] / 255) * (hdr ? hdrScale : 1),
        w: draft.w,
      },
      true,
    );
  };

  const setHsvChannel = (index: number, next: number) => {
    if (index === 0) {
      emit(applyHdrScale(hsvToLinearColor({ ...hsv, h: next }, draft.w)), true);
      return;
    }
    if (index === 1) {
      emit(applyHdrScale(hsvToLinearColor({ ...hsv, s: next / 100 }, draft.w)), true);
      return;
    }
    if (index === 2) {
      emit(applyHdrScale(hsvToLinearColor({ ...hsv, v: next / 100 }, draft.w)), true);
      return;
    }

    const normalized = hsvToLinearColor(hsv, draft.w);
    emit(
      {
        x: normalized.x * next,
        y: normalized.y * next,
        z: normalized.z * next,
        w: draft.w,
      },
      true,
    );
  };

  const channelSliders =
    mode === 'hsv'
      ? [
          {
            label: 'H',
            value: hsv.h,
            min: 0,
            max: 360,
            step: 0.1,
            precision: 1,
            className: 'is-hue',
            onChange: (next: number) => setHsvChannel(0, next),
          },
          {
            label: 'S',
            value: hsv.s * 100,
            min: 0,
            max: 100,
            step: 0.1,
            precision: 1,
            className: 'is-saturation',
            onChange: (next: number) => setHsvChannel(1, next),
          },
          {
            label: 'V',
            value: hsv.v * 100,
            min: 0,
            max: 100,
            step: 0.1,
            precision: 1,
            className: 'is-value',
            onChange: (next: number) => setHsvChannel(2, next),
          },
          ...(hdr
            ? [
                {
                  label: 'Intensity',
                  value: hdrScale,
                  min: 1,
                  max: Math.max(1, maxChannelValue),
                  step: 0.01,
                  precision: 2,
                  className: 'is-intensity',
                  onChange: (next: number) => setHsvChannel(3, next),
                },
              ]
            : []),
          ...(showAlpha
            ? [
                {
                  label: 'A',
                  value: draft.w,
                  min: 0,
                  max: 1,
                  step: 0.001,
                  precision: 3,
                  className: 'is-alpha',
                  onChange: (next: number) => emit({ ...draft, w: clamp(next) }, true),
                },
              ]
            : []),
        ]
      : [
          {
            label: 'R',
            value: mode === 'linear' ? draft.x : linearToSrgb(draft.x / hdrScale) * 255,
            min: mode === 'linear' ? minChannelValue : 0,
            max: mode === 'linear' ? maxChannelValue : 255,
            step: mode === 'linear' ? 0.001 : 1,
            precision: mode === 'linear' ? 3 : 0,
            className: 'is-red',
            onChange: (next: number) => setRgbChannel(0, next),
          },
          {
            label: 'G',
            value: mode === 'linear' ? draft.y : linearToSrgb(draft.y / hdrScale) * 255,
            min: mode === 'linear' ? minChannelValue : 0,
            max: mode === 'linear' ? maxChannelValue : 255,
            step: mode === 'linear' ? 0.001 : 1,
            precision: mode === 'linear' ? 3 : 0,
            className: 'is-green',
            onChange: (next: number) => setRgbChannel(1, next),
          },
          {
            label: 'B',
            value: mode === 'linear' ? draft.z : linearToSrgb(draft.z / hdrScale) * 255,
            min: mode === 'linear' ? minChannelValue : 0,
            max: mode === 'linear' ? maxChannelValue : 255,
            step: mode === 'linear' ? 0.001 : 1,
            precision: mode === 'linear' ? 3 : 0,
            className: 'is-blue',
            onChange: (next: number) => setRgbChannel(2, next),
          },
          ...(showAlpha
            ? [
                {
                  label: 'A',
                  value: draft.w,
                  min: 0,
                  max: 1,
                  step: 0.001,
                  precision: 3,
                  className: 'is-alpha',
                  onChange: (next: number) => setRgbChannel(3, next),
                },
              ]
            : []),
        ];

  return createPortal(
    <UiDialog
      ariaLabel={`${label} color picker`}
      blurBackdrop={false}
      className="arc-color-picker"
      compact
      footer={
        <>
          <UiButton onClick={cancel} type="button">
            Cancel
          </UiButton>
          <UiButton onClick={accept} type="button" variant="primary">
            OK
          </UiButton>
        </>
      }
      initialPosition={initialPosition}
      modal={false}
      onClose={onClose}
      showCloseButton={false}
      title={label}
      width={pickerWidth}
      zIndex={1600}
    >
      <div className="arc-color-picker-layout">
        <section className="arc-color-picker-wheel-pane">
          <div className="arc-color-picker-preset">
            <UiDropdown
              ariaLabel="Color preset"
              className="arc-color-preset-dropdown"
              onValueChange={() => undefined}
              options={colorPresetOptions}
              value="custom"
            />
            <UiIconButton
              className="arc-color-eyedropper"
              disabled={!eyeDropper}
              label="Pick color from screen"
              onClick={() => {
                if (!eyeDropper) return;
                void new eyeDropper().open().then((result) => {
                  const sampled = hexToLinearColor(result.sRGBHex, draft.w);
                  if (sampled) emit(sampled, true);
                });
              }}
              title={eyeDropper ? 'Pick an sRGB color from the screen' : 'Screen eyedropper is unavailable'}
            >
              <Pipette size={16} />
            </UiIconButton>
          </div>

          <div className="arc-color-wheel-wrap">
            <div
              aria-label="Hue"
              className="arc-color-wheel"
              onPointerDown={(event) => huePointer(event, false)}
              onPointerMove={(event) => huePointer(event, false)}
              onPointerUp={(event) => huePointer(event, true)}
            >
              <span className="arc-color-wheel-cursor" style={hueCursorStyle} />
              <div
                aria-label="Saturation and value"
                className="arc-color-wheel-square"
                onPointerDown={(event) => spectrumPointer(event, false)}
                onPointerMove={(event) => spectrumPointer(event, false)}
                onPointerUp={(event) => spectrumPointer(event, true)}
                style={{ '--arc-picker-hue': hueCss } as CSSProperties}
              >
                <span
                  className="arc-color-spectrum-cursor"
                  style={{ left: `${hsv.s * 100}%`, top: `${(1 - hsv.v) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </section>

        <section className="arc-color-picker-controls-pane">
          <div className="arc-color-preview-stack">
            <button
              aria-label={`Restore original ${label}`}
              className="arc-color-preview-row is-original"
              onClick={() => emit(original.current, true)}
              type="button"
            >
              <small>Original</small>
              <span className="arc-color-preview-swatch">
                <i style={swatchStyle(originalCss)} />
              </span>
            </button>
            <div className="arc-color-preview-row is-current">
              <small>Current</small>
              <span className="arc-color-preview-swatch">
                <i style={swatchStyle(currentCss)} />
              </span>
            </div>
          </div>

          <div className="arc-color-picker-mode">
            <UiDropdown
              ariaLabel="Color representation"
              className="arc-color-mode-dropdown"
              onValueChange={setMode}
              options={colorModeOptions}
              value={mode}
            />
          </div>

          <div
            className="arc-color-channel-sliders"
            style={
              {
                '--arc-picker-hue': hueCss,
                '--arc-picker-color': colorToCss({ ...draft, w: 1 }),
              } as CSSProperties
            }
          >
            {channelSliders.map((channel) => (
              <PickerChannelSlider
                className={channel.className}
                key={`${mode}-${channel.label}`}
                label={channel.label}
                max={channel.max}
                min={channel.min}
                precision={channel.precision}
                step={channel.step}
                value={channel.value}
                onChange={channel.onChange}
              />
            ))}
          </div>

          <div className="arc-color-hex-row">
            <label htmlFor="arc-color-hex">Hex sRGB</label>
            <PickerTextField
              id="arc-color-hex"
              value={colorToHex(draft, showAlpha)}
              onCommit={(hex) => {
                const parsed = hexToLinearColor(hex, draft.w);
                if (parsed) emit(parsed, true);
              }}
            />
            <UiIconButton
              className="arc-color-copy"
              label="Copy color hex"
              onClick={() => void navigator.clipboard?.writeText(colorToHex(draft, showAlpha))}
            >
              <Copy size={14} />
            </UiIconButton>
          </div>
        </section>
      </div>
    </UiDialog>,
    document.body,
  );
}

function PickerChannelSlider({
  className,
  label,
  value,
  precision,
  min,
  max,
  step,
  onChange,
}: {
  className: string;
  label: string;
  value: number;
  precision: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className={`arc-color-channel-slider ${className}`}>
      <span>{label}</span>
      <input
        aria-label={`Color ${label}`}
        max={max}
        min={min}
        onChange={(event) => onChange(event.target.valueAsNumber)}
        step={step}
        type="range"
        value={value}
      />
      <output>{value.toFixed(precision)}</output>
    </label>
  );
}

function PickerTextField({ id, value, onCommit }: { id: string; value: string; onCommit: (value: string) => void }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return (
    <input
      aria-label="Hex sRGB"
      id={id}
      onBlur={() => onCommit(text)}
      onChange={(event) => setText(event.target.value)}
      onFocus={(event) => event.currentTarget.select()}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur();
        if (event.key === 'Escape') {
          setText(value);
          event.currentTarget.blur();
        }
      }}
      spellCheck={false}
      value={text}
    />
  );
}
