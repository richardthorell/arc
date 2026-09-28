// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ArcAndroidBridge } from '../../../../common/androidDeviceTypes';
import { LevelEditorToolbar } from './LevelEditorToolbar';

afterEach(() => {
  cleanup();
  delete window.arcAndroid;
});

describe('LevelEditorToolbar Android targets', () => {
  it('discovers ADB devices and exposes model, API level, and connection state', async () => {
    const listDevices = vi.fn<ArcAndroidBridge['listDevices']>().mockResolvedValue({
      available: true,
      adbPath: 'C:\\Android\\Sdk\\platform-tools\\adb.exe',
      adbVersion: '1.0.41',
      devices: [
        {
          serial: 'SERIAL1',
          state: 'online',
          stateDetail: 'device',
          model: 'Pixel 9 Pro',
          product: 'komodo',
          device: 'komodo',
          transportId: '1',
          apiLevel: 36,
          emulator: false,
        },
        {
          serial: 'SERIAL2',
          state: 'unauthorized',
          stateDetail: 'unauthorized',
          model: 'Pixel 7',
          product: 'panther',
          device: 'panther',
          transportId: '2',
          emulator: false,
        },
      ],
      error: '',
      refreshedAt: '2026-09-28T08:00:00.000Z',
    });
    window.arcAndroid = { listDevices };

    render(
      <LevelEditorToolbar
        configuredTargetPlatforms={['android']}
        onCommand={vi.fn()}
        targetPlatform="android"
      />,
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('Pixel 9 Pro'));
    expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('API 36');
    expect(listDevices).toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Target device' }));
    expect(screen.getByRole('option', { name: /Pixel 9 Pro.*API 36/ })).toBeEnabled();
    expect(screen.getByRole('option', { name: /Pixel 7.*Unauthorized/ })).toBeDisabled();
  });

  it('shows ADB as unavailable when the configured SDK cannot provide it', async () => {
    window.arcAndroid = {
      listDevices: vi.fn().mockResolvedValue({
        available: false,
        adbPath: '',
        adbVersion: '',
        devices: [],
        error: 'Android SDK / ADB is not configured',
        refreshedAt: '2026-09-28T08:00:00.000Z',
      }),
    };

    render(
      <LevelEditorToolbar
        configuredTargetPlatforms={['android']}
        onCommand={vi.fn()}
        targetPlatform="android"
      />,
    );

    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('ADB unavailable'));
    expect(screen.getByRole('button', { name: 'Target device' })).toBeEnabled();
  });
});
