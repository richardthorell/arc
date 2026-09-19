import { useLayoutEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';
import { Link2, RotateCcw } from 'lucide-react';

import type { Vec3 } from './inspectorTypes';
import type { NumberFieldSchema, VectorAxis, Vector3FieldSchema } from './propertySchema';

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
  mixed?: boolean;
  onPreview?: (value: number) => void;
  onCommit: (value: number) => void;
};

const clamp = (value: number, min?: number, max?: number) =>
  Math.min(Math.max(value, min ?? -Infinity), max ?? Infinity);
const formatNumber = (value: number, precision: number) => Number(value).toFixed(precision);

export function NumericInput({
  ariaLabel,
  value,
  precision,
  step,
  scrubSensitivity,
  unit,
  min,
  max,
  scrubLabel,
  scrubClassName,
  mixed = false,
  onPreview,
  onCommit,
}: NumericInputProps) {
  const [draft, setDraft] = useState(() => (mixed ? '' : formatNumber(value, precision)));
  const [scrubbing, setScrubbing] = useState(false);
  const cancelBlur = useRef(false);
  const latestScrub = useRef(value);
  const frame = useRef<number | null>(null);

  useLayoutEffect(() => setDraft(mixed ? '' : formatNumber(value, precision)), [mixed, precision, value]);

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
        className={mixed ? 'is-mixed' : undefined}
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
  field,
  value,
  linked,
  onToggleLinked,
  onReset,
  onPreview,
  onCommit,
  mixed = false,
  showLabel = true,
}: {
  field: Pick<Vector3FieldSchema, 'label' | 'precision' | 'step' | 'scrubSensitivity' | 'unit' | 'linked' | 'tooltip'>;
  value: Vec3;
  linked: boolean;
  onToggleLinked?: () => void;
  onReset?: () => void;
  onPreview: (axis: VectorAxis, value: number) => void;
  onCommit: (axis: VectorAxis, value: number) => void;
  mixed?: boolean;
  showLabel?: boolean;
}) {
  const linkButton = field.linked ? (
    <button
      aria-label={`${linked ? 'Unlink' : 'Link'} ${field.label.toLocaleLowerCase()} axes`}
      className={`inspector-scale-link ${linked ? 'is-linked' : ''}`}
      onClick={onToggleLinked}
      title={`${linked ? 'Unlink' : 'Link'} scale axes`}
      type="button"
    >
      <Link2 aria-hidden="true" size={13} strokeWidth={2} />
    </button>
  ) : null;

  return (
    <div
      className={showLabel ? 'inspector-property inspector-vector-property' : 'inspector-vector-control'}
      title={field.tooltip}
    >
      {showLabel && (
        <div className="inspector-property-label">
          <span>{field.label}</span>
          {linkButton}
        </div>
      )}
      <div className="inspector-vector-value">
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
              mixed={mixed}
              onCommit={(next) => onCommit(axis, next)}
              onPreview={(next) => onPreview(axis, next)}
            />
          ))}
        </div>
        {(!showLabel && linkButton) || onReset ? (
          <div className="inspector-vector-actions">
            {!showLabel && linkButton}
            {onReset && (
              <button
                aria-label={`Reset ${field.label}`}
                className="inspector-field-reset"
                onClick={onReset}
                title={`Reset ${field.label}`}
                type="button"
              >
                <RotateCcw aria-hidden="true" size={12} />
              </button>
            )}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function NumberControlLabel({
  field,
  value,
  onPreview,
  onCommit,
}: {
  field: Pick<NumberFieldSchema, 'label' | 'scrubSensitivity' | 'min' | 'max'>;
  value: number;
  onPreview: (value: number) => void;
  onCommit: (value: number) => void;
}) {
  const latestScrub = useRef(value);
  const frame = useRef<number | null>(null);

  const startScrub = (event: ReactPointerEvent<HTMLSpanElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startX = event.clientX;
    const startValue = value;
    latestScrub.current = value;

    const flushPreview = () => {
      frame.current = null;
      onPreview(latestScrub.current);
    };
    const move = (moveEvent: PointerEvent) => {
      latestScrub.current = clamp(
        startValue + (moveEvent.clientX - startX) * field.scrubSensitivity,
        field.min,
        field.max,
      );
      if (frame.current === null) frame.current = window.requestAnimationFrame(flushPreview);
    };
    const finish = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', finish);
      if (frame.current !== null) {
        window.cancelAnimationFrame(frame.current);
        frame.current = null;
      }
      onCommit(latestScrub.current);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', finish, { once: true });
  };

  return (
    <span className="inspector-number-row-label" onPointerDown={startScrub}>
      {field.label}
    </span>
  );
}

export function NumberControl({
  field,
  value,
  onPreview,
  onCommit,
  mixed = false,
  showLabel = true,
}: {
  field: Pick<NumberFieldSchema, 'label' | 'precision' | 'step' | 'scrubSensitivity' | 'unit' | 'min' | 'max'>;
  value: number;
  onPreview: (value: number) => void;
  onCommit: (value: number) => void;
  mixed?: boolean;
  showLabel?: boolean;
}) {
  const input = (
    <NumericInput
      ariaLabel={field.label}
      max={field.max}
      min={field.min}
      precision={field.precision}
      scrubClassName={showLabel ? 'inspector-scalar-scrub' : undefined}
      scrubLabel={showLabel ? field.label : undefined}
      scrubSensitivity={field.scrubSensitivity}
      step={field.step}
      unit={field.unit}
      value={value}
      mixed={mixed}
      onCommit={onCommit}
      onPreview={onPreview}
    />
  );
  return showLabel ? <div className="inspector-property inspector-number-property">{input}</div> : input;
}
