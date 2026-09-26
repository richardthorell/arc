// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { PlaySessionBar } from './PlaySessionBar';

afterEach(cleanup);

describe('PlaySessionBar', () => {
  it('keeps an active paused session visible and controllable outside the level toolbar', () => {
    const onCommand = vi.fn();
    render(<PlaySessionBar state="paused" tickId={42} onCommand={onCommand} />);

    expect(screen.getByText('Play World')).toBeInTheDocument();
    expect(screen.getByText('Paused')).toBeInTheDocument();
    expect(screen.getByText('Tick 42')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Resume Play World' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Pause Play World' })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'Step Play World' }));
    fireEvent.click(screen.getByRole('button', { name: 'Stop Play World' }));
    expect(onCommand).toHaveBeenNthCalledWith(1, 'scene.step');
    expect(onCommand).toHaveBeenNthCalledWith(2, 'scene.stop');
  });

  it('surfaces a fault and limits recovery to stopping the session', () => {
    render(<PlaySessionBar state="faulted" error="Game system failed" tickId={8} onCommand={vi.fn()} />);

    expect(screen.getByRole('alert')).toHaveTextContent('Game system failed');
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Step Play World' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Stop Play World' })).toBeEnabled();
  });
});
