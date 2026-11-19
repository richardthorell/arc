// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { requestedSettingsDialogKind, resetSettingsDialogRequest } from '../settings/settingsDialogRoute';
import { configuredTargetPlatformsForProject, MainToolbar } from './MainToolbar';

afterEach(() => {
  cleanup();
  localStorage.clear();
  resetSettingsDialogRequest();
});

describe('MainToolbar runtime controls', () => {
  it('disables unconfigured platforms and refreshes availability when settings change', () => {
    const onTargetPlatformChange = vi.fn();
    render(
      <MainToolbar
        configuredTargetPlatforms={['windows', 'android']}
        onCommand={vi.fn()}
        onTargetPlatformChange={onTargetPlatformChange}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Play platform' }));
    expect(screen.getByRole('option', { name: /Android/ })).toBeDisabled();
    fireEvent.click(screen.getByRole('option', { name: /Android/ }));
    expect(onTargetPlatformChange).not.toHaveBeenCalled();
    void act(() =>
      window.dispatchEvent(
        new CustomEvent('arc-editor-settings-changed', {
          detail: {
            pathValidation: {
              'platform.android.javaHome': { valid: true },
              'platform.android.sdkPath': { valid: true },
              'platform.android.ndkPath': { valid: true },
            },
          },
        }),
      ),
    );
    expect(screen.getByRole('option', { name: /Android/ })).toBeEnabled();
    expect(screen.getByRole('option', { name: /Windows/ })).toBeDisabled();
    fireEvent.click(screen.getByRole('option', { name: /Android/ }));
    expect(onTargetPlatformChange).toHaveBeenCalledWith('android');
  });
  it('renders host-authoritative playback state and enables only valid controls', () => {
    const { rerender } = render(<MainToolbar onCommand={vi.fn()} runtimeState="stopped" timeScale={2} />);

    expect(screen.getByRole('button', { name: 'Pause' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Step' })).toBeDisabled();
    expect(screen.queryByTestId('toolbar-runtime-state')).not.toBeInTheDocument();

    rerender(<MainToolbar onCommand={vi.fn()} runtimeState="running" timeScale={2} />);
    expect(screen.getByRole('button', { name: 'Play' })).toHaveClass('is-active');
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pause' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Stop' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Step' })).toBeDisabled();
    expect(screen.queryByTestId('toolbar-runtime-state')).not.toBeInTheDocument();

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

  it('keeps only platform and Play-target selectors on the right and opens project platform settings', () => {
    const onCommand = vi.fn();
    const onTargetPlatformChange = vi.fn();
    const onTargetDeviceChange = vi.fn();
    const { rerender } = render(
      <MainToolbar
        configuredTargetPlatforms={['windows', 'android']}
        onCommand={onCommand}
        onTargetDeviceChange={onTargetDeviceChange}
        onTargetPlatformChange={onTargetPlatformChange}
        targetDevice="pixel-9"
        targetDevices={[{ id: 'pixel-9', label: 'Pixel 9', platform: 'android' }]}
        targetPlatform="android"
      />,
    );
    expect(screen.getByRole('button', { name: 'Play platform' })).toHaveTextContent('Android');
    expect(screen.getByRole('button', { name: 'Play target' })).toHaveTextContent('Pixel 9');
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Play' })).toHaveAttribute(
      'title',
      expect.stringContaining('Remote Play'),
    );
    expect(screen.queryByRole('button', { name: 'Build Editor Code' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open build dashboard' })).not.toBeInTheDocument();
    expect(screen.queryByText('Authoring World')).not.toBeInTheDocument();
    expect(screen.queryByText('Play World: Running')).not.toBeInTheDocument();

    void act(() =>
      window.dispatchEvent(
        new CustomEvent('arc-editor-settings-changed', {
          detail: {
            pathValidation: Object.fromEntries(
              [
                'platform.windows.msvcToolchainPath',
                'platform.windows.sdkPath',
                'platform.windows.cmakePath',
                'platform.windows.ninjaPath',
                'platform.android.javaHome',
                'platform.android.sdkPath',
                'platform.android.ndkPath',
              ].map((key) => [key, { valid: true }]),
            ),
          },
        }),
      ),
    );

    fireEvent.click(screen.getByRole('button', { name: 'Play platform' }));
    expect(screen.getByRole('option', { name: /Windows/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /Android/ })).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Linux/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: /Xbox/ })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('option', { name: /Platform Settings/ }));
    expect(requestedSettingsDialogKind()).toBe('projectSettings');
    expect(onCommand).toHaveBeenCalledWith('settings.open');

    rerender(
      <MainToolbar
        configuredTargetPlatforms={['windows', 'android']}
        onCommand={onCommand}
        targetPlatform="windows"
      />,
    );
    expect(screen.getByRole('button', { name: 'Play target' })).toHaveTextContent('Editor Viewport');
    expect(screen.getByRole('button', { name: 'Play' })).toBeEnabled();
  });

  it('defaults to the editor viewport, remembers local selections, and never launches remotely via Play', () => {
    const onCommand = vi.fn();
    const { rerender } = render(
      <MainToolbar configuredTargetPlatforms={['windows', 'android']} onCommand={onCommand} targetPlatform="windows" />,
    );
    const play = screen.getByRole('button', { name: 'Play' });
    expect(play).toBeEnabled();
    fireEvent.click(play);
    expect(onCommand).toHaveBeenCalledWith('scene.play');

    fireEvent.click(screen.getByRole('button', { name: 'Play target' }));
    fireEvent.click(screen.getByRole('option', { name: /This Computer \(Standalone\)/ }));
    expect(screen.getByRole('button', { name: 'Play target' })).toHaveTextContent('This Computer (Standalone)');
    expect(play).toBeDisabled();
    expect(localStorage.getItem('arc.editor.play-targets.v1')).toContain('standalone');

    rerender(
      <MainToolbar
        configuredTargetPlatforms={['windows', 'android']}
        onCommand={onCommand}
        targetPlatform="android"
        targetDevices={[{ id: 'device:pixel', label: 'Pixel 9', platform: 'android' }]}
      />,
    );
    expect(screen.getByRole('button', { name: 'Play target' })).toHaveTextContent('Select a target');
    fireEvent.click(screen.getByRole('button', { name: 'Play target' }));
    fireEvent.click(screen.getByRole('option', { name: 'Pixel 9' }));
    expect(screen.getByRole('button', { name: 'Play target' })).toHaveTextContent('Pixel 9');
    expect(play).toBeDisabled();

    rerender(<MainToolbar configuredTargetPlatforms={['windows', 'android']} onCommand={onCommand} targetPlatform="windows" />);
    expect(screen.getByRole('button', { name: 'Play target' })).toHaveTextContent('This Computer (Standalone)');
    fireEvent.click(screen.getByRole('button', { name: 'Play target' }));
    fireEvent.click(screen.getByRole('option', { name: 'Editor Viewport' }));
    expect(play).toBeEnabled();
    expect(onCommand).toHaveBeenCalledTimes(1);
  });

  it('retains a missing device choice instead of automatically selecting a different target', () => {
    render(
      <MainToolbar
        configuredTargetPlatforms={['android']}
        onCommand={vi.fn()}
        targetPlatform="android"
        targetDevice="device:missing"
        targetDevices={[{ id: 'device:available', label: 'Pixel 9', platform: 'android' }]}
      />,
    );
    expect(screen.getByRole('button', { name: 'Play target' })).toHaveTextContent('Selected target unavailable');
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'Play target' }));
    expect(screen.getByRole('option', { name: 'Pixel 9' })).toBeEnabled();
  });

  it('locks Play destination changes while a local runtime is active', () => {
    const { rerender } = render(
      <MainToolbar onCommand={vi.fn()} runtimeState="running" targetPlatform="windows" />,
    );
    expect(screen.getByRole('button', { name: 'Play platform' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Play target' })).toBeDisabled();
    rerender(<MainToolbar onCommand={vi.fn()} runtimeState="stopped" targetPlatform="windows" />);
    expect(screen.getByRole('button', { name: 'Play target' })).toBeEnabled();
  });

});
