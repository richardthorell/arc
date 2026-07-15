import { useEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import { Link2 } from 'lucide-react';

import type { ColorChannel, NumberFieldSchema, VectorAxis, Vector3FieldSchema } from './componentSchemas';
import { colorToCss, ColorPicker } from './ColorPicker';
import type { Vec3, Vec4 } from './inspectorTypes';

type NumericInputProps = {
  ariaLabel: string;
  value: number;
  precision: number;
  step: number;
  scrubSensitivity: number;
  unit?: string;
  min?: number;
  max?: number;
  scrubLabel?: string;
  scrubClassName?: string;
  onPreview?: (value: number) => void;
  onCommit: (value: number) => void;
};

const clamp = (value: number, min?: number, max?: number) => Math.min(Math.max(value, min ?? -Infinity), max ?? Infinity);
const formatNumber = (value: number, precision: number) => Number(value).toFixed(precision);

export function NumericInput({
  ariaLabel, value, precision, step, scrubSensitivity, unit, min, max, scrubLabel,
  scrubClassName, onPreview, onCommit,
}: NumericInputProps) {
  const [draft, setDraft] = useState(() => formatNumber(value, precision));
  const [scrubbing, setScrubbing] = useState(false);
  const cancelBlur = useRef(false);
  const latestScrub = useRef(value);
  const frame = useRef<number | null>(null);

  useEffect(() => setDraft(formatNumber(value, precision)), [precision, value]);

  const commitDraft = () => {
    if (cancelBlur.current) {
      cancelBlur.current = false;
      return;
    }
    const parsed = Number.parseFloat(draft);
    if (!Number.isFinite(parsed)) {
      setDraft(formatNumber(value, precision));
      return;
    }
    const next = clamp(parsed, min, max);
    setDraft(formatNumber(next, precision));
    onCommit(next);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      commitDraft();
      cancelBlur.current = true;
      event.currentTarget.blur();
    } else if (event.key === 'Escape') {
      cancelBlur.current = true;
      setDraft(formatNumber(value, precision));
      event.currentTarget.blur();
    } else if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
      const parsed = Number.parseFloat(draft);
      const base = Number.isFinite(parsed) ? parsed : value;
      const multiplier = event.shiftKey ? 10 : event.altKey ? 0.1 : 1;
      const next = clamp(base + (event.key === 'ArrowUp' ? step : -step) * multiplier, min, max);
      setDraft(formatNumber(next, precision));
      onCommit(next);
    }
  };

  const startScrub = (event: ReactPointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startX = event.clientX;
    const startValue = value;
    latestScrub.current = value;
    setScrubbing(true);

    const flushPreview = () => {
      frame.current = null;
      onPreview?.(latestScrub.current);
    };
    const move = (moveEvent: PointerEvent) => {
      latestScrub.current = clamp(startValue + (moveEvent.clientX - startX) * scrubSensitivity, min, max);
      setDraft(formatNumber(latestScrub.current, precision));
      if (frame.current === null) {
        frame.current = window.requestAnimationFrame(flushPreview);
      }
    };
    const finish = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', finish);
      if (frame.current !== null) {
        window.cancelAnimationFrame(frame.current);
        frame.current = null;
      }
      setScrubbing(false);
      onCommit(latestScrub.current);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', finish, { once: true });
  };

  return (
    <span className={`inspector-number ${scrubbing ? 'is-scrubbing' : ''}`}>
      {scrubLabel && (
        <span
          aria-hidden="true"
          className={`inspector-number-scrub ${scrubClassName ?? ''}`}
          onPointerDown={startScrub}
        >
          {scrubLabel}
        </span>
      )}
      <input
        aria-label={ariaLabel}
        inputMode="decimal"
        onBlur={commitDraft}
        onChange={(event) => setDraft(event.target.value)}
        onFocus={(event) => event.currentTarget.select()}
        onKeyDown={onKeyDown}
        value={draft}
      />
      {unit && <span className="inspector-number-unit">{unit}</span>}
    </span>
  );
}

export function Vector3Control({
  field, value, linked, onToggleLinked, onPreview, onCommit,
}: {
  field: Vector3FieldSchema;
  value: Vec3;
  linked: boolean;
  onToggleLinked?: () => void;
  onPreview: (axis: VectorAxis, value: number) => void;
  onCommit: (axis: VectorAxis, value: number) => void;
}) {
  return (
    <div className="inspector-property inspector-vector-property">
      <div className="inspector-property-label">
        <span>{field.label}</span>
        {field.linked && (
          <button
            aria-label={`${linked ? 'Unlink' : 'Link'} ${field.label.toLocaleLowerCase()} axes`}
            className={`inspector-scale-link ${linked ? 'is-linked' : ''}`}
            onClick={onToggleLinked}
            title="Link scale axes"
            type="button"
          >
            <Link2 aria-hidden="true" size={13} strokeWidth={2} />
          </button>
        )}
      </div>
      <div className="inspector-axis-grid">
        {(['x', 'y', 'z'] as const).map((axis) => (
          <NumericInput
            key={axis}
            ariaLabel={`${field.label} ${axis.toUpperCase()}`}
            precision={field.precision}
            scrubClassName={`axis-${axis}`}
            scrubLabel={axis.toUpperCase()}
            scrubSensitivity={field.scrubSensitivity}
            step={field.step}
            unit={field.unit}
            value={value[axis]}
            onCommit={(next) => onCommit(axis, next)}
            onPreview={(next) => onPreview(axis, next)}
          />
        ))}
      </div>
    </div>
  );
}

export function NumberControl({ field, value, onPreview, onCommit }: {
  field: NumberFieldSchema;
  value: number;
  onPreview: (value: number) => void;
  onCommit: (value: number) => void;
}) {
  return (
    <div className="inspector-property inspector-number-property">
      <NumericInput
        ariaLabel={field.label}
        max={field.max}
        min={field.min}
        precision={field.precision}
        scrubClassName="inspector-scalar-scrub"
        scrubLabel={field.label}
        scrubSensitivity={field.scrubSensitivity}
        step={field.step}
        unit={field.unit}
        value={value}
        onCommit={onCommit}
        onPreview={onPreview}
      />
    </div>
  );
}

export function ColorControl({ label, value, onPreview, onCommit }: {
  label: string;
  value: Vec4;
  onPreview: (value: Vec4) => void;
  onCommit: (value: Vec4) => void;
}) {
  const [pickerOpen, setPickerOpen] = useState(false);
  const swatchRef = useRef<HTMLButtonElement>(null);
  const update = (channel: ColorChannel, next: number) => onCommit({ ...value, [channel]: clamp(next, 0, 1) });
  return (
    <div className="inspector-property inspector-color-property">
      <span className="inspector-property-label">{label}</span>
      <div className="inspector-color-control">
        <button
          aria-expanded={pickerOpen}
          aria-label={`Open ${label} color picker`}
          className="inspector-color-swatch"
          onClick={() => setPickerOpen((open) => !open)}
          ref={swatchRef}
          title="Open the advanced linear color picker"
          type="button"
        ><span style={{ background: colorToCss(value) }} /></button>
        {(['x', 'y', 'z', 'w'] as const).map((channel, index) => (
          <NumericInput
            key={channel}
            ariaLabel={`${label} ${'RGBA'[index]}`}
            max={1}
            min={0}
            precision={2}
            scrubSensitivity={0.005}
            step={0.01}
            value={value[channel]}
            onCommit={(next) => update(channel, next)}
          />
        ))}
      </div>
      {pickerOpen && <ColorPicker anchorRef={swatchRef} label={label} value={value}
        onClose={() => setPickerOpen(false)} onCommit={onCommit} onPreview={onPreview} />}
    </div>
  );
}
