import { useEffect, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent, PointerEvent as ReactPointerEvent, RefObject } from 'react';
import { Check, Copy, Pipette } from 'lucide-react';
import { createPortal } from 'react-dom';

import { UiButton } from './UiButton';
import { UiDialog } from './UiDialog';

import './UiColorPicker.css';

export type UiColorValue = { x: number; y: number; z: number; w: number };

type HsvColor = { h: number; s: number; v: number };
type ColorMode = 'rgb' | 'hsv';
type ColorSpace = 'srgb' | 'linear';

type EyeDropperResult = { sRGBHex: string };
type EyeDropperInstance = { open: () => Promise<EyeDropperResult> };
type EyeDropperConstructor = new () => EyeDropperInstance;

export type UiColorPickerProps = {
  anchorRef: RefObject<HTMLElement | null>;
  label: string;
  showAlpha?: boolean;
  minChannelValue?: number;
  maxChannelValue?: number;
  value: UiColorValue;
  onClose: () => void;
  onCommit: (value: UiColorValue) => void;
  onPreview: (value: UiColorValue) => void;
};

const pickerWidth = 306;
const pickerEstimatedHeight = 500;
const pickerViewportMargin = 8;

const anchoredPickerPosition = (anchorRef: RefObject<HTMLElement | null>) => {
  const anchor = anchorRef.current?.getBoundingClientRect();
  if (!anchor) return { x: pickerViewportMargin, y: pickerViewportMargin };

  const x = Math.max(
    pickerViewportMargin,
    Math.min(anchor.left, window.innerWidth - pickerWidth - pickerViewportMargin),
  );
  let y = anchor.bottom + 6;
  if (y + pickerEstimatedHeight > window.innerHeight) {
    y = Math.max(pickerViewportMargin, anchor.top - pickerEstimatedHeight - 6);
  }
  return { x, y };
};

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
  anchorRef,
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
  const [space, setSpace] = useState<ColorSpace>('srgb');
  const original = useRef(value);
  const latest = useRef(value);
  const previewFrame = useRef<number | null>(null);
  const pendingPreview = useRef(value);
  const hsv = linearColorToHsv(draft);
  const hdr = maxChannelValue > 1;
  const initialPosition = anchoredPickerPosition(anchorRef);
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
    if (final) {
      if (previewFrame.current !== null) {
        window.cancelAnimationFrame(previewFrame.current);
        previewFrame.current = null;
      }
      onCommit(next);
      return;
    }
    if (previewFrame.current === null) {
      previewFrame.current = window.requestAnimationFrame(() => {
        previewFrame.current = null;
        onPreview(pendingPreview.current);
      });
    }
  };

  const updateSpectrum = (event: ReactPointerEvent<HTMLDivElement>, final: boolean) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const saturation = clamp((event.clientX - bounds.left) / Math.max(bounds.width, 1));
    const brightness = 1 - clamp((event.clientY - bounds.top) / Math.max(bounds.height, 1));
    emit(applyHdrScale(hsvToLinearColor({ h: hsv.h, s: saturation, v: brightness }, draft.w)), final);
  };

  const spectrumPointer = (event: ReactPointerEvent<HTMLDivElement>, final: boolean) => {
    if (event.type === 'pointerdown') event.currentTarget.setPointerCapture(event.pointerId);
    if (event.type === 'pointermove' && !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    updateSpectrum(event, final);
    if (final && event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const setHue = (hue: number, final: boolean) =>
    emit(applyHdrScale(hsvToLinearColor({ ...hsv, h: hue }, draft.w)), final);
  const setAlpha = (alpha: number, final: boolean) => emit({ ...draft, w: clamp(alpha) }, final);
  const currentCss = colorToCss(draft);
  const originalCss = colorToCss(original.current);
  const hueCss = colorToCss(hsvToLinearColor({ h: hsv.h, s: 1, v: 1 }, 1));
  const swatchStyle = (color: string) => ({ '--arc-picker-color': color }) as CSSProperties;

  const finish = (value: UiColorValue) => {
    if (previewFrame.current !== null) {
      window.cancelAnimationFrame(previewFrame.current);
      previewFrame.current = null;
    }
    onCommit(value);
    onClose();
  };
  const cancel = () => finish(original.current);
  const accept = () => finish(latest.current);

  const commitChannels = (channels: number[]) => {
    const alpha = showAlpha ? channels[3] : draft.w;
    if (mode === 'hsv') {
      emit(
        applyHdrScale(hsvToLinearColor({ h: channels[0], s: channels[1] / 100, v: channels[2] / 100 }, alpha)),
        true,
      );
      return;
    }
    if (space === 'linear') {
      emit({ x: channels[0], y: channels[1], z: channels[2], w: alpha }, true);
      return;
    }
    emit(
      {
        x: srgbToLinear(channels[0] / 255) * (hdr ? hdrScale : 1),
        y: srgbToLinear(channels[1] / 255) * (hdr ? hdrScale : 1),
        z: srgbToLinear(channels[2] / 255) * (hdr ? hdrScale : 1),
        w: alpha,
      },
      true,
    );
  };

  const channels =
    mode === 'hsv'
      ? [hsv.h, hsv.s * 100, hsv.v * 100, ...(showAlpha ? [draft.w] : [])]
      : space === 'linear'
        ? [draft.x, draft.y, draft.z, ...(showAlpha ? [draft.w] : [])]
        : [
            linearToSrgb(draft.x / hdrScale) * 255,
            linearToSrgb(draft.y / hdrScale) * 255,
            linearToSrgb(draft.z / hdrScale) * 255,
            ...(showAlpha ? [draft.w] : []),
          ];
  const channelLabels =
    mode === 'hsv' ? ['H', 'S', 'V', ...(showAlpha ? ['A'] : [])] : ['R', 'G', 'B', ...(showAlpha ? ['A'] : [])];

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
      <div className="arc-color-picker-preview-row">
        <button
          aria-label={`Restore original ${label}`}
          className="arc-color-preview"
          onClick={() => emit(original.current, true)}
          style={swatchStyle(originalCss)}
          type="button"
        >
          <span />
          <small>Original</small>
        </button>
        <div className="arc-color-preview is-current" style={swatchStyle(currentCss)}>
          <span />
          <small>Current</small>
        </div>
        <button
          aria-label="Copy color hex"
          className="arc-color-tool"
          onClick={() => void navigator.clipboard?.writeText(colorToHex(draft, showAlpha))}
          title="Copy sRGB hexadecimal value"
          type="button"
        >
          <Copy size={14} />
        </button>
        <button
          aria-label="Pick color from screen"
          className="arc-color-tool"
          disabled={!eyeDropper}
          onClick={() => {
            if (!eyeDropper) return;
            void new eyeDropper().open().then((result) => {
              const sampled = hexToLinearColor(result.sRGBHex, draft.w);
              if (sampled) emit(sampled, true);
            });
          }}
          title={eyeDropper ? 'Pick an sRGB color from the screen' : 'Screen eyedropper is unavailable'}
          type="button"
        >
          <Pipette size={15} />
        </button>
      </div>

      <div
        aria-label="Saturation and value"
        className="arc-color-spectrum"
        onPointerDown={(event) => spectrumPointer(event, false)}
        onPointerMove={(event) => spectrumPointer(event, false)}
        onPointerUp={(event) => spectrumPointer(event, true)}
        style={{ '--arc-picker-hue': hueCss } as CSSProperties}
      >
        <span className="arc-color-spectrum-cursor" style={{ left: `${hsv.s * 100}%`, top: `${(1 - hsv.v) * 100}%` }} />
      </div>

      <PickerRange
        label="Hue"
        className="arc-color-hue"
        min={0}
        max={360}
        step={0.1}
        value={hsv.h}
        onChange={(next) => setHue(next, false)}
        onFinal={() => setHue(linearColorToHsv(latest.current).h, true)}
      />
      {showAlpha && (
        <PickerRange
          label="Alpha"
          className="arc-color-alpha"
          min={0}
          max={1}
          step={0.001}
          value={draft.w}
          style={{ '--arc-picker-color': colorToCss({ ...draft, w: 1 }) } as CSSProperties}
          onChange={(next) => setAlpha(next, false)}
          onFinal={() => setAlpha(latest.current.w, true)}
        />
      )}

      <div className="arc-color-picker-options">
        <div className="arc-color-segments" aria-label="Color model">
          {(['rgb', 'hsv'] as const).map((option) => (
            <button
              className={mode === option ? 'is-active' : ''}
              key={option}
              onClick={() => setMode(option)}
              type="button"
            >
              {option.toUpperCase()}
            </button>
          ))}
        </div>
        <div className="arc-color-segments" aria-label="RGB color space">
          {(['srgb', 'linear'] as const).map((option) => (
            <button
              className={space === option ? 'is-active' : ''}
              disabled={mode === 'hsv'}
              key={option}
              onClick={() => setSpace(option)}
              type="button"
            >
              {option === 'srgb' ? 'sRGB' : 'Linear'}
            </button>
          ))}
        </div>
      </div>

      <div
        className="arc-color-channel-grid"
        style={{ gridTemplateColumns: `repeat(${channels.length}, minmax(0, 1fr))` }}
      >
        {channels.map((channel, index) => (
          <PickerNumberField
            key={`${mode}-${space}-${channelLabels[index]}`}
            label={channelLabels[index]}
            max={
              mode === 'hsv'
                ? index === 0
                  ? 360
                  : index === 3
                    ? 1
                    : 100
                : space === 'srgb' && index < 3
                  ? 255
                  : index < 3
                    ? maxChannelValue
                    : 1
            }
            min={mode === 'rgb' && space === 'linear' && index < 3 ? minChannelValue : 0}
            precision={mode === 'hsv' ? (index === 0 ? 1 : index === 3 ? 3 : 1) : space === 'srgb' && index < 3 ? 0 : 3}
            value={channel}
            onCommit={(next) => {
              const updated = [...channels];
              updated[index] = next;
              commitChannels(updated);
            }}
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
        <span title="Values are converted to ARC's scene-linear color storage">
          <Check size={13} /> Linear storage
        </span>
      </div>
    </UiDialog>,
    document.body,
  );
}

function PickerRange({
  label,
  className,
  value,
  min,
  max,
  step,
  style,
  onChange,
  onFinal,
}: {
  label: string;
  className: string;
  value: number;
  min: number;
  max: number;
  step: number;
  style?: CSSProperties;
  onChange: (value: number) => void;
  onFinal: () => void;
}) {
  return (
    <label className="arc-color-range">
      <span>{label}</span>
      <input
        aria-label={label}
        className={className}
        max={max}
        min={min}
        onChange={(event) => onChange(event.target.valueAsNumber)}
        onKeyUp={onFinal}
        onPointerUp={onFinal}
        step={step}
        style={style}
        type="range"
        value={value}
      />
    </label>
  );
}

function PickerNumberField({
  label,
  value,
  precision,
  min,
  max,
  onCommit,
}: {
  label: string;
  value: number;
  precision: number;
  min: number;
  max: number;
  onCommit: (value: number) => void;
}) {
  const [text, setText] = useState(value.toFixed(precision));
  useEffect(() => setText(value.toFixed(precision)), [precision, value]);
  const commit = () => {
    const parsed = Number.parseFloat(text);
    if (!Number.isFinite(parsed)) return setText(value.toFixed(precision));
    onCommit(clamp(parsed, min, max));
  };
  return (
    <label>
      <span>{label}</span>
      <input
        aria-label={`Color ${label}`}
        inputMode="decimal"
        onBlur={commit}
        onChange={(event) => setText(event.target.value)}
        onFocus={(event) => event.currentTarget.select()}
        onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
          if (event.key === 'Enter') event.currentTarget.blur();
          if (event.key === 'Escape') {
            setText(value.toFixed(precision));
            event.currentTarget.blur();
          }
        }}
        value={text}
      />
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
