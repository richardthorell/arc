// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { UiSlider } from './UiSlider';

describe('UiSlider', () => {
  it('exposes range semantics and emits numeric values', () => {
    const onValueChange = vi.fn();
    render(
      <UiSlider
        aria-label="Zoom"
        max={180}
        min={35}
        step={5}
        value={100}
        onValueChange={onValueChange}
      />,
    );

    const slider = screen.getByRole('slider', { name: 'Zoom' });
    expect(slider).toHaveAttribute('min', '35');
    expect(slider).toHaveAttribute('max', '180');
    expect(slider).toHaveAttribute('step', '5');
    expect(slider).toHaveValue('100');

    fireEvent.change(slider, { target: { value: '125' } });
    expect(onValueChange).toHaveBeenCalledWith(125);
  });

  it('publishes the fill progress as a css custom property', () => {
    render(<UiSlider aria-label="Blend" max={1} min={0} value={0.25} onValueChange={() => undefined} />);

    expect(screen.getByRole('slider', { name: 'Blend' })).toHaveStyle({
      '--ui-slider-progress': '25%',
    });
  });
});
