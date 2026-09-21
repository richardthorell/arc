import { useRef } from 'react';
import type { PointerEvent as ReactPointerEvent } from 'react';
import { UiNumericInput } from '../ui/UiNumericInput';
import type { NumberFieldSchema } from './propertySchema';

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
    <UiNumericInput
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
