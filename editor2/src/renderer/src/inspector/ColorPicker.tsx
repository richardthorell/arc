import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { CSSProperties, KeyboardEvent, PointerEvent as ReactPointerEvent, RefObject } from 'react';
import { Check, Copy, Pipette, X } from 'lucide-react';
import { createPortal } from 'react-dom';

import type { Vec4 } from './inspectorTypes';

type HsvColor = { h: number; s: number; v: number };
type ColorMode = 'rgb' | 'hsv';
type ColorSpace = 'srgb' | 'linear';

type EyeDropperResult = { sRGBHex: string };
type EyeDropperInstance = { open: () => Promise<EyeDropperResult> };
type EyeDropperConstructor = new () => EyeDropperInstance;

export type ColorPickerProps = {
  anchorRef: RefObject<HTMLElement | null>;
  label: string;
  showAlpha?: boolean;
  value: Vec4;
  onClose: () => void;
  onCommit: (value: Vec4) => void;
  onPreview: (value: Vec4) => void;
};

const pickerWidth = 306;
const pickerEstimatedHeight = 500;
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

const byteHex = (value: number) => Math.round(clamp(value) * 255).toString(16).padStart(2, '0').toUpperCase();

export const colorToHex = (value: Vec4, includeAlpha = true) => {
  const rgb = `${byteHex(linearToSrgb(value.x))}${byteHex(linearToSrgb(value.y))}${byteHex(linearToSrgb(value.z))}`;
  return `#${rgb}${includeAlpha ? byteHex(value.w) : ''}`;
};

export const hexToLinearColor = (hex: string, fallbackAlpha = 1): Vec4 | null => {
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

export const linearColorToHsv = (value: Vec4): HsvColor => {
  const r = linearToSrgb(value.x);
  const g = linearToSrgb(value.y);
  const b = linearToSrgb(value.z);
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

export const hsvToLinearColor = (hsv: HsvColor, alpha: number): Vec4 => {
  const hue = wrapHue(hsv.h);
  const saturation = clamp(hsv.s);
  const value = clamp(hsv.v);
  const chroma = value * saturation;
  const x = chroma * (1 - Math.abs((hue / 60) % 2 - 1));
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

export const colorToCss = (value: Vec4) =>
  `rgba(${Math.round(linearToSrgb(value.x) * 255)}, ${Math.round(linearToSrgb(value.y) * 255)}, ${Math.round(linearToSrgb(value.z) * 255)}, ${clamp(value.w)})`;

export function ColorPicker({ anchorRef, label, value, showAlpha = true, onClose, onCommit, onPreview }: ColorPickerProps) {
  const [draft, setDraft] = useState(value);
  const [mode, setMode] = useState<ColorMode>('rgb');
  const [space, setSpace] = useState<ColorSpace>('srgb');
  const [position, setPosition] = useState({ left: 8, top: 8 });
  const original = useRef(value);
  const latest = useRef(value);
  const pickerRef = useRef<HTMLDivElement>(null);
  const previewFrame = useRef<number | null>(null);
  const pendingPreview = useRef(value);
  const hsv = linearColorToHsv(draft);
  const eyeDropper = (window as unknown as { EyeDropper?: EyeDropperConstructor }).EyeDropper;

  useEffect(() => {
    setDraft(value);
    latest.current = value;
  }, [value]);

  useLayoutEffect(() => {
    const anchor = anchorRef.current?.getBoundingClientRect();
    if (!anchor) return;
    let left = Math.min(anchor.left, window.innerWidth - pickerWidth - 8);
    let top = anchor.bottom + 6;
    if (top + pickerEstimatedHeight > window.innerHeight) top = Math.max(8, anchor.top - pickerEstimatedHeight - 6);
    setPosition({ left: Math.max(8, left), top });
  }, [anchorRef]);

  useEffect(() => {
    const closeFromOutside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!pickerRef.current?.contains(target) && !anchorRef.current?.contains(target)) onClose();
    };
    const closeFromKeyboard = (event: globalThis.KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      onCommit(original.current);
      onClose();
    };
    document.addEventListener('pointerdown', closeFromOutside, true);
    document.addEventListener('keydown', closeFromKeyboard);
    return () => {
      document.removeEventListener('pointerdown', closeFromOutside, true);
      document.removeEventListener('keydown', closeFromKeyboard);
      if (previewFrame.current !== null) window.cancelAnimationFrame(previewFrame.current);
    };
  }, [anchorRef, onClose, onCommit]);

  const emit = (next: Vec4, final: boolean) => {
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
    emit(hsvToLinearColor({ h: hsv.h, s: saturation, v: brightness }, draft.w), final);
  };

  const spectrumPointer = (event: ReactPointerEvent<HTMLDivElement>, final: boolean) => {
    if (event.type === 'pointerdown') event.currentTarget.setPointerCapture(event.pointerId);
    if (event.type === 'pointermove' && !event.currentTarget.hasPointerCapture(event.pointerId)) return;
    updateSpectrum(event, final);
    if (final && event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const setHue = (hue: number, final: boolean) => emit(hsvToLinearColor({ ...hsv, h: hue }, draft.w), final);
  const setAlpha = (alpha: number, final: boolean) => emit({ ...draft, w: clamp(alpha) }, final);
  const currentCss = colorToCss(draft);
  const originalCss = colorToCss(original.current);
  const hueCss = colorToCss(hsvToLinearColor({ h: hsv.h, s: 1, v: 1 }, 1));
  const swatchStyle = (color: string) => ({ '--arc-picker-color': color } as CSSProperties);

  const commitChannels = (channels: number[]) => {
    const alpha = showAlpha ? channels[3] : draft.w;
    if (mode === 'hsv') {
      emit(hsvToLinearColor({ h: channels[0], s: channels[1] / 100, v: channels[2] / 100 }, alpha), true);
      return;
    }
    if (space === 'linear') {
      emit({ x: channels[0], y: channels[1], z: channels[2], w: alpha }, true);
      return;
    }
    emit({
      x: srgbToLinear(channels[0] / 255), y: srgbToLinear(channels[1] / 255),
      z: srgbToLinear(channels[2] / 255), w: alpha,
    }, true);
  };

  const channels = mode === 'hsv'
    ? [hsv.h, hsv.s * 100, hsv.v * 100, ...(showAlpha ? [draft.w] : [])]
    : space === 'linear'
      ? [draft.x, draft.y, draft.z, ...(showAlpha ? [draft.w] : [])]
      : [linearToSrgb(draft.x) * 255, linearToSrgb(draft.y) * 255, linearToSrgb(draft.z) * 255, ...(showAlpha ? [draft.w] : [])];
  const channelLabels = mode === 'hsv' ? ['H', 'S', 'V', ...(showAlpha ? ['A'] : [])] : ['R', 'G', 'B', ...(showAlpha ? ['A'] : [])];

  return createPortal(
    <div
      aria-label={`${label} color picker`}
      aria-modal="false"
      className="arc-color-picker"
      ref={pickerRef}
      role="dialog"
      style={{ left: position.left, top: position.top }}
    >
      <header className="arc-color-picker-header">
        <strong>{label}</strong>
        <span>{showAlpha ? 'Linear RGBA' : 'Linear RGB'}</span>
        <button aria-label="Close color picker" onClick={onClose} type="button"><X size={14} /></button>
      </header>

      <div className="arc-color-picker-preview-row">
        <button aria-label={`Restore original ${label}`} className="arc-color-preview" onClick={() => emit(original.current, true)}
          style={swatchStyle(originalCss)} type="button"><span /><small>Original</small></button>
        <div className="arc-color-preview is-current" style={swatchStyle(currentCss)}><span /><small>Current</small></div>
        <button
          aria-label="Copy color hex"
          className="arc-color-tool"
          onClick={() => void navigator.clipboard?.writeText(colorToHex(draft, showAlpha))}
          title="Copy sRGB hexadecimal value"
          type="button"
        ><Copy size={14} /></button>
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
        ><Pipette size={15} /></button>
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

      <PickerRange label="Hue" className="arc-color-hue" min={0} max={360} step={0.1} value={hsv.h}
        onChange={(next) => setHue(next, false)} onFinal={() => setHue(linearColorToHsv(latest.current).h, true)} />
      {showAlpha && <PickerRange label="Alpha" className="arc-color-alpha" min={0} max={1} step={0.001} value={draft.w}
        style={{ '--arc-picker-color': colorToCss({ ...draft, w: 1 }) } as CSSProperties}
        onChange={(next) => setAlpha(next, false)} onFinal={() => setAlpha(latest.current.w, true)} />}

      <div className="arc-color-picker-options">
        <div className="arc-color-segments" aria-label="Color model">
          {(['rgb', 'hsv'] as const).map((option) => <button className={mode === option ? 'is-active' : ''} key={option}
            onClick={() => setMode(option)} type="button">{option.toUpperCase()}</button>)}
        </div>
        <div className="arc-color-segments" aria-label="RGB color space">
          {(['srgb', 'linear'] as const).map((option) => <button className={space === option ? 'is-active' : ''} disabled={mode === 'hsv'} key={option}
            onClick={() => setSpace(option)} type="button">{option === 'srgb' ? 'sRGB' : 'Linear'}</button>)}
        </div>
      </div>

      <div className="arc-color-channel-grid" style={{ gridTemplateColumns: `repeat(${channels.length}, minmax(0, 1fr))` }}>
        {channels.map((channel, index) => (
          <PickerNumberField
            key={`${mode}-${space}-${channelLabels[index]}`}
            label={channelLabels[index]}
            max={mode === 'hsv' ? (index === 0 ? 360 : index === 3 ? 1 : 100) : space === 'srgb' && index < 3 ? 255 : 1}
            min={0}
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
        <PickerTextField id="arc-color-hex" value={colorToHex(draft, showAlpha)} onCommit={(hex) => {
          const parsed = hexToLinearColor(hex, draft.w);
          if (parsed) emit(parsed, true);
        }} />
        <span title="Values are converted to ARC's scene-linear color storage"><Check size={13} /> Linear storage</span>
      </div>
    </div>,
    document.body,
  );
}

function PickerRange({ label, className, value, min, max, step, style, onChange, onFinal }: {
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
  return <label className="arc-color-range"><span>{label}</span><input aria-label={label} className={className} max={max} min={min}
    onChange={(event) => onChange(event.target.valueAsNumber)} onKeyUp={onFinal} onPointerUp={onFinal} step={step} style={style} type="range" value={value} /></label>;
}

function PickerNumberField({ label, value, precision, min, max, onCommit }: {
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
  return <label><span>{label}</span><input aria-label={`Color ${label}`} inputMode="decimal" onBlur={commit}
    onChange={(event) => setText(event.target.value)} onFocus={(event) => event.currentTarget.select()}
    onKeyDown={(event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter') event.currentTarget.blur();
      if (event.key === 'Escape') { setText(value.toFixed(precision)); event.currentTarget.blur(); }
    }} value={text} /></label>;
}

function PickerTextField({ id, value, onCommit }: { id: string; value: string; onCommit: (value: string) => void }) {
  const [text, setText] = useState(value);
  useEffect(() => setText(value), [value]);
  return <input aria-label="Hex sRGB" id={id} onBlur={() => onCommit(text)} onChange={(event) => setText(event.target.value)}
    onFocus={(event) => event.currentTarget.select()} onKeyDown={(event) => {
      if (event.key === 'Enter') event.currentTarget.blur();
      if (event.key === 'Escape') { setText(value); event.currentTarget.blur(); }
    }} spellCheck={false} value={text} />;
}
