// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { MainToolbar } from './MainToolbar';

afterEach(cleanup);

describe('MainToolbar runtime controls', () => {
  it('renders host-authoritative playback state', () => {
    render(<MainToolbar onCommand={vi.fn()} runtimeState="running" timeScale={2} />);

    expect(screen.getByRole('button', { name: 'Play' })).toHaveClass('is-active');
    expect(screen.getByRole('button', { name: 'Pause' })).not.toHaveClass('is-active');
    expect(screen.getByText('2×')).toBeInTheDocument();
  });

  it('dispatches playback commands and time-scale changes', () => {
    const onCommand = vi.fn();
    const onCycleTimeScale = vi.fn();
    render(<MainToolbar onCommand={onCommand} runtimeState="paused"
      timeScale={0.5} onCycleTimeScale={onCycleTimeScale} />);

    fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    fireEvent.click(screen.getByRole('button', { name: 'Step' }));
    fireEvent.click(screen.getByText('0.5×'));

    expect(onCommand).toHaveBeenNthCalledWith(1, 'scene.play');
    expect(onCommand).toHaveBeenNthCalledWith(2, 'scene.step');
    expect(onCycleTimeScale).toHaveBeenCalledOnce();
  });
});
