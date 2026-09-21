import { useLayoutEffect, useRef, useState } from 'react';
import type { KeyboardEvent, PointerEvent as ReactPointerEvent } from 'react';

import './UiNumericInput.css';

export type UiNumericInputProps = {
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

export function UiNumericInput({
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
}: UiNumericInputProps) {
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
      if (frame.current === null) frame.current = window.requestAnimationFrame(flushPreview);
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
    <span className={`ui-numeric-input ${scrubbing ? 'is-scrubbing' : ''}`}>
      {scrubLabel && (
        <span
          aria-hidden="true"
          className={['ui-numeric-input-scrub', scrubClassName].filter(Boolean).join(' ')}
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
      {unit && <span className="ui-numeric-input-unit">{unit}</span>}
    </span>
  );
}
