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
    render(
      <MainToolbar onCommand={onCommand} runtimeState="paused" timeScale={0.5} onCycleTimeScale={onCycleTimeScale} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Play' }));
    fireEvent.click(screen.getByRole('button', { name: 'Step' }));
    fireEvent.click(screen.getByText('0.5×'));

    expect(onCommand).toHaveBeenNthCalledWith(1, 'scene.play');
    expect(onCommand).toHaveBeenNthCalledWith(2, 'scene.step');
    expect(onCycleTimeScale).toHaveBeenCalledOnce();
  });

  it('uses labeled dropdowns for transform origin, coordinate space, and snap increments', () => {
    const onCoordinateSpaceChange = vi.fn();
    const onRotationSnapChange = vi.fn();
    const onTranslationSnapChange = vi.fn();
    const onScaleSnapChange = vi.fn();
    render(
      <MainToolbar
        coordinateSpace="world"
        onCommand={vi.fn()}
        onCoordinateSpaceChange={onCoordinateSpaceChange}
        onRotationSnapChange={onRotationSnapChange}
        onTranslationSnapChange={onTranslationSnapChange}
        onScaleSnapChange={onScaleSnapChange}
        rotationSnap={15}
        translationSnap={0.25}
        scaleSnap={0.25}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Transform origin' }));
    fireEvent.click(screen.getByRole('option', { name: /Center/ }));
    expect(screen.getByRole('button', { name: 'Transform origin' })).toHaveTextContent('Center');

    fireEvent.click(screen.getByRole('button', { name: 'Coordinate space' }));
    fireEvent.click(screen.getByRole('option', { name: 'Local' }));
    expect(onCoordinateSpaceChange).toHaveBeenCalledWith('local');

    expect(screen.getByRole('button', { name: 'Rotation snap' })).toHaveTextContent('Rotate 15°');
    expect(screen.getByRole('button', { name: 'Translation snap' })).toHaveTextContent('Move 0.25');
    expect(screen.getByRole('button', { name: 'Scale snap' })).toHaveTextContent('Scale 25%');

    fireEvent.click(screen.getByRole('button', { name: 'Rotation snap' }));
    fireEvent.click(screen.getByRole('option', { name: 'Rotate 45°' }));
    expect(onRotationSnapChange).toHaveBeenCalledWith(45);

    fireEvent.click(screen.getByRole('button', { name: 'Translation snap' }));
    fireEvent.click(screen.getByRole('option', { name: 'Move 1' }));
    expect(onTranslationSnapChange).toHaveBeenCalledWith(1);

    fireEvent.click(screen.getByRole('button', { name: 'Scale snap' }));
    fireEvent.click(screen.getByRole('option', { name: 'Scale 50%' }));
    expect(onScaleSnapChange).toHaveBeenCalledWith(0.5);
  });

  it('uses platform and split-build controls instead of layout and windows buttons', () => {
    const onTargetPlatformChange = vi.fn();
    const onBuildAction = vi.fn();
    render(
      <MainToolbar
        onCommand={vi.fn()}
        onBuildAction={onBuildAction}
        onTargetPlatformChange={onTargetPlatformChange}
        targetPlatform="linux"
      />,
    );

    expect(screen.queryByText('Layouts')).not.toBeInTheDocument();
    expect(screen.queryByText('Windows', { selector: 'summary' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Target platform' })).toHaveTextContent('Linux');
    expect(
      screen.getByRole('button', { name: 'Target platform' }).querySelector('[data-platform-icon="linux"]'),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Target platform' }));
    for (const platform of ['Windows', 'Linux', 'macOS', 'iOS', 'Android', 'Xbox', 'PlayStation', 'Nintendo Switch']) {
      expect(screen.getByRole('option', { name: new RegExp(platform) })).toBeInTheDocument();
    }
    for (const platform of ['windows', 'linux', 'macos', 'ios', 'android', 'xbox', 'playstation', 'switch']) {
      expect(document.querySelector(`[data-platform-icon="${platform}"]`)).toBeInTheDocument();
    }

    fireEvent.click(screen.getByRole('option', { name: /Xbox/ }));
    expect(onTargetPlatformChange).toHaveBeenCalledWith('xbox');

    fireEvent.click(screen.getByRole('button', { name: 'Build' }));
    expect(onBuildAction).toHaveBeenCalledWith('build');

    fireEvent.click(screen.getByRole('button', { name: 'Build actions' }));
    fireEvent.click(screen.getByRole('menuitem', { name: /Rebuild/ }));
    expect(onBuildAction).toHaveBeenCalledWith('rebuild');
  });
});
