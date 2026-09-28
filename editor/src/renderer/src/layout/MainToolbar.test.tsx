// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { requestedSettingsDialogKind, resetSettingsDialogRequest } from '../settings/settingsDialogRoute';
import { configuredTargetPlatformsForProject, MainToolbar } from './MainToolbar';

afterEach(() => {
  cleanup();
  resetSettingsDialogRequest();
});

describe('MainToolbar runtime controls', () => {
  it('renders host-authoritative playback state and enables only valid controls', () => {
    const { rerender } = render(<MainToolbar onCommand={vi.fn()} runtimeState="stopped" timeScale={2} />);

    expect(screen.getByRole('button', { name: 'Pause' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Step' })).toBeDisabled();
    expect(screen.getByTestId('toolbar-runtime-state')).toHaveTextContent('Authoring World');

    rerender(<MainToolbar onCommand={vi.fn()} runtimeState="running" timeScale={2} />);
    expect(screen.getByRole('button', { name: 'Play' })).toHaveClass('is-active');
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pause' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Step' })).toBeDisabled();
    expect(screen.getByTestId('toolbar-runtime-state')).toHaveTextContent('Play World: Running');

    rerender(<MainToolbar onCommand={vi.fn()} runtimeState="paused" timeScale={2} />);
    expect(screen.getByRole('button', { name: 'Resume' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Pause' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Step' })).toBeEnabled();
  });

  it('dispatches playback commands and changes time scale from playback options', () => {
    const onCommand = vi.fn();
    const onTimeScaleChange = vi.fn();
    render(
      <MainToolbar onCommand={onCommand} runtimeState="paused" timeScale={0.5} onTimeScaleChange={onTimeScaleChange} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Resume' }));
    fireEvent.click(screen.getByRole('button', { name: 'Step' }));
    expect(onCommand).toHaveBeenNthCalledWith(1, 'scene.play');
    expect(onCommand).toHaveBeenNthCalledWith(2, 'scene.step');

    expect(screen.queryByText('0.5×')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Playback options' }));
    expect(screen.getByRole('menu', { name: 'Playback options menu' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Simulation time scale' }));
    fireEvent.click(screen.getByRole('option', { name: '2×' }));
    expect(onTimeScaleChange).toHaveBeenCalledWith(2);
  });

  it('uses compact transform dropdowns and a grouped snap settings menu', () => {
    const onCoordinateSpaceChange = vi.fn();
    const onToggleSnapping = vi.fn();
    const onRotationSnapChange = vi.fn();
    const onTranslationSnapChange = vi.fn();
    const onScaleSnapChange = vi.fn();
    render(
      <MainToolbar
        coordinateSpace="world"
        onCommand={vi.fn()}
        onCoordinateSpaceChange={onCoordinateSpaceChange}
        onToggleSnapping={onToggleSnapping}
        onRotationSnapChange={onRotationSnapChange}
        onTranslationSnapChange={onTranslationSnapChange}
        onScaleSnapChange={onScaleSnapChange}
        rotationSnap={15}
        translationSnap={0.25}
        scaleSnap={0.25}
        snapping
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Transform origin' }));
    fireEvent.click(screen.getByRole('option', { name: /Center/ }));
    expect(screen.getByRole('button', { name: 'Transform origin' })).toHaveTextContent('Center');

    fireEvent.click(screen.getByRole('button', { name: 'Coordinate space' }));
    fireEvent.click(screen.getByRole('option', { name: 'Local' }));
    expect(onCoordinateSpaceChange).toHaveBeenCalledWith('local');

    expect(screen.queryByRole('button', { name: 'Rotation snap' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Translation snap' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Scale snap' })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Snap settings' }));
    expect(screen.getByRole('menu', { name: 'Snap settings menu' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Enable snapping' }));
    expect(onToggleSnapping).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole('button', { name: 'Rotation snap' }));
    fireEvent.click(screen.getByRole('option', { name: '45°' }));
    expect(onRotationSnapChange).toHaveBeenCalledWith(45);

    fireEvent.click(screen.getByRole('button', { name: 'Translation snap' }));
    fireEvent.click(screen.getByRole('option', { name: '1' }));
    expect(onTranslationSnapChange).toHaveBeenCalledWith(1);

    fireEvent.click(screen.getByRole('button', { name: 'Scale snap' }));
    fireEvent.click(screen.getByRole('option', { name: '50%' }));
    expect(onScaleSnapChange).toHaveBeenCalledWith(0.5);
  });

  it('derives enabled editor platforms from project targets and falls back to the host', () => {
    expect(
      configuredTargetPlatformsForProject(
        {
          targetPlatforms: [
            { id: 'windows-x64-vulkan', enabled: true },
            { id: 'android-arm64-vulkan', enabled: true },
            { id: 'linux-x64-vulkan', enabled: false },
          ],
          cookProfiles: [
            {
              id: 'windows-x64-vulkan',
              platform: 'windows',
              architecture: 'x86_64',
              renderer: 'vulkan',
              api: '1.2',
              textures: { outputs: ['bc'], quality: 'balanced' },
              configuration: 'Shipping',
            },
            {
              id: 'android-arm64-vulkan',
              platform: 'android',
              architecture: 'arm64',
              renderer: 'vulkan',
              api: '1.2',
              textures: { outputs: ['astc'], quality: 'balanced' },
              configuration: 'Shipping',
            },
          ],
        },
        'windows',
      ),
    ).toEqual(['windows', 'android']);
    expect(configuredTargetPlatformsForProject({ targetPlatforms: [], cookProfiles: [] }, 'macos')).toEqual(['macos']);
  });

  it('shows only project platforms, opens platform settings, and exposes the target device next to it', () => {
    const onCommand = vi.fn();
    const onTargetPlatformChange = vi.fn();
    const onTargetDeviceChange = vi.fn();
    const onBuildAction = vi.fn();
    render(
      <MainToolbar
        configuredTargetPlatforms={['windows', 'android']}
        onCommand={onCommand}
        onBuildAction={onBuildAction}
        onTargetDeviceChange={onTargetDeviceChange}
        onTargetPlatformChange={onTargetPlatformChange}
        targetDevice="pixel-9"
        targetDevices={[
          { id: 'pixel-9', label: 'Pixel 9', platform: 'android' },
          { id: 'local-windows', label: 'This Computer', platform: 'windows' },
        ]}
        targetPlatform="android"
      />,
    );

    expect(screen.getByRole('button', { name: 'Target platform' })).toHaveTextContent('Android');
    expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('Pixel 9');

    fireEvent.click(screen.getByRole('button', { name: 'Target platform' }));
    expect(screen.getByRole('option', { name: /Windows/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /Android/ })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Linux/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Xbox/ })).not.toBeInTheDocument();
    expect(screen.getByRole('separator')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('option', { name: /Platform Settings/ }));
    expect(requestedSettingsDialogKind()).toBe('projectSettings');
    expect(onCommand).toHaveBeenCalledWith('settings.open');

    fireEvent.click(screen.getByRole('button', { name: 'Target platform' }));
    fireEvent.click(screen.getByRole('option', { name: /Windows/ }));
    expect(onTargetPlatformChange).toHaveBeenCalledWith('windows');

    fireEvent.click(screen.getByRole('button', { name: 'Target device' }));
    fireEvent.click(screen.getByRole('option', { name: /Pixel 9/ }));
    expect(onTargetDeviceChange).toHaveBeenCalledWith('pixel-9');

    fireEvent.click(screen.getByRole('button', { name: 'Build' }));
    expect(onBuildAction).toHaveBeenCalledWith('build');

    fireEvent.click(screen.getByRole('button', { name: 'Build actions' }));
    fireEvent.click(screen.getByRole('menuitem', { name: /Rebuild/ }));
    expect(onBuildAction).toHaveBeenCalledWith('rebuild');
  });
});
