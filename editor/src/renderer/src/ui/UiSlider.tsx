import type { CSSProperties, InputHTMLAttributes } from 'react';

import './UiSlider.css';

export type UiSliderProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  'max' | 'min' | 'onChange' | 'step' | 'type' | 'value'
> & {
  min: number;
  max: number;
  step?: number;
  value: number;
  onValueChange: (value: number) => void;
};

export function UiSlider({ min, max, step = 1, value, onValueChange, className, style, ...props }: UiSliderProps) {
  const range = Math.max(Number.EPSILON, max - min);
  const progress = Math.min(1, Math.max(0, (value - min) / range));
  const sliderStyle = {
    ...style,
    '--ui-slider-progress': `${progress * 100}%`,
  } as CSSProperties;

  return (
    <input
      {...props}
      className={['ui-slider', className].filter(Boolean).join(' ')}
      max={max}
      min={min}
      onChange={(event) => onValueChange(Number(event.target.value))}
      step={step}
      style={sliderStyle}
      type="range"
      value={value}
    />
  );
}
