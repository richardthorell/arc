// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { act, cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiLabContentCards } from './UiLabContentCards';

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe('UiLabContentCards', () => {
  it('shows texture, material, and model Content Browser cards', () => {
    const view = render(<UiLabContentCards />);

    expect(view.getAllByRole('option')).toHaveLength(3);
    expect(view.getByText('Texture')).toBeVisible();
    expect(view.getByText('Material')).toBeVisible();
    expect(view.getByText('Model')).toBeVisible();
    expect(view.getByText(/Hover any card/)).toBeVisible();
  });

  it('keeps the production hover details enabled in the lab', async () => {
    vi.useFakeTimers();
    const view = render(<UiLabContentCards />);
    const textureCard = view.getAllByRole('option')[0];

    fireEvent.mouseEnter(textureCard, { clientX: 120, clientY: 90 });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(350);
    });

    const tooltip = view.getByRole('tooltip');
    expect(tooltip).toHaveTextContent('T_Mountain_Sunset');
    expect(tooltip).toHaveTextContent('2048 × 2048');
  });
});
