// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { AndroidTargetSnapshot } from '../../../../common/androidDeviceTypes';
import { useAndroidTargets, androidPreferenceKey } from '../../app/useAndroidTargets';
import { LevelEditorToolbar } from './LevelEditorToolbar';

const snapshot = (): AndroidTargetSnapshot => ({
  sdkRoot: '/sdk',
  available: true,
  adbPath: '/sdk/adb',
  adbVersion: '36',
  devices: [],
  error: '',
  refreshedAt: '',
  emulatorAvailable: true,
  diagnostics: [],
  targets: [
    {
      id: 'device:phone',
      name: 'Pixel 9',
      kind: 'device',
      serial: 'phone',
      apiLevel: 36,
      state: 'online',
      ready: true,
      canStart: false,
    },
    {
      id: 'device:locked',
      name: 'Pixel 9',
      kind: 'device',
      serial: 'locked',
      state: 'unauthorized',
      ready: false,
      canStart: false,
    },
    {
      id: 'avd:Tablet',
      name: 'Tablet',
      kind: 'emulator',
      avdName: 'Tablet',
      state: 'stopped',
      ready: false,
      canStart: true,
    },
  ],
});
function Harness({
  platform = 'android',
  project = '/project',
}: {
  platform?: 'android' | 'windows';
  project?: string;
}) {
  const targets = useAndroidTargets(platform === 'android', project);
  return (
    <LevelEditorToolbar
      androidTargets={targets}
      configuredTargetPlatforms={['android', 'windows']}
      targetPlatform={platform}
      onCommand={vi.fn()}
    />
  );
}
afterEach(() => {
  cleanup();
  delete window.arcAndroid;
  localStorage.clear();
  vi.useRealTimers();
});
function bridge(value = snapshot()) {
  const result = {
    listDevices: vi.fn(),
    listTargets: vi.fn().mockResolvedValue(value),
    startEmulator: vi.fn().mockResolvedValue(undefined),
    resolveTarget: vi.fn(),
  };
  window.arcAndroid = result;
  return result;
}
describe('Android target picker', () => {
  it('polls every two seconds only while active and keeps the choice across platform switches', async () => {
    vi.useFakeTimers();
    const api = bridge();
    const view = renderHook(({ active }) => useAndroidTargets(active, '/project'), { initialProps: { active: true } });
    await act(async () => {
      await Promise.resolve();
    });
    expect(api.listTargets).toHaveBeenCalledTimes(1);
    act(() => view.result.current.select('avd:Tablet'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });
    expect(api.listTargets).toHaveBeenCalledTimes(3);
    view.rerender({ active: false });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(4000);
    });
    expect(api.listTargets).toHaveBeenCalledTimes(3);
    view.rerender({ active: true });
    await act(async () => {
      await Promise.resolve();
    });
    expect(view.result.current.selected?.id).toBe('avd:Tablet');
  });
  it('selects the sole ready target, labels duplicate names with serials, and disables unauthorized devices', async () => {
    bridge();
    render(<Harness />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('phone'));
    expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('API 36');
    fireEvent.click(screen.getByRole('button', { name: 'Target device' }));
    expect(screen.getByRole('option', { name: /Pixel 9.*phone/ })).toBeEnabled();
    expect(screen.getByRole('option', { name: /Pixel 9.*locked/ })).toBeDisabled();
    expect(screen.getByRole('option', { name: /Tablet.*stopped/ })).toBeEnabled();
  });
  it('selects an AVD without launching, persists it, and starts only on explicit action', async () => {
    const api = bridge();
    render(<Harness />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('phone'));
    fireEvent.click(screen.getByRole('button', { name: 'Target device' }));
    fireEvent.click(screen.getByRole('option', { name: /Tablet/ }));
    expect(api.startEmulator).not.toHaveBeenCalled();
    expect(localStorage.getItem(androidPreferenceKey('/project', '/sdk'))).toBe('avd:Tablet');
    fireEvent.click(screen.getByRole('button', { name: 'Start Emulator' }));
    await waitFor(() => expect(api.startEmulator).toHaveBeenCalledWith({ sdkRoot: '/sdk', id: 'avd:Tablet' }));
  });
  it('restores a missing preference without substituting another ready device, then reconnects', async () => {
    localStorage.setItem(androidPreferenceKey('/project', '/sdk'), 'device:missing');
    const value = snapshot();
    const api = bridge(value);
    render(<Harness />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('Selected target unavailable'),
    );
    api.listTargets.mockResolvedValue({
      ...value,
      targets: [...value.targets, { ...value.targets[0], id: 'device:missing', name: 'Reconnected' }],
    });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh Android targets' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('Reconnected'));
  });
  it('requires a choice when multiple targets are ready and retains a disconnected choice', async () => {
    const value = snapshot();
    value.targets[1] = { ...value.targets[1], state: 'online', ready: true };
    const api = bridge(value);
    render(<Harness />);
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('Select Android target'),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Target device' }));
    fireEvent.click(screen.getByRole('option', { name: /Pixel 9.*phone/ }));
    api.listTargets.mockResolvedValue({ ...value, targets: value.targets.slice(1) });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh Android targets' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('Selected target unavailable'),
    );
  });
  it('reports diagnostics without hiding connected devices', async () => {
    const value = snapshot();
    value.diagnostics = ['Emulator missing. Install it in Android Studio.'];
    bridge(value);
    render(<Harness />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('phone'));
    expect(screen.getByRole('status')).toHaveTextContent('Emulator missing');
  });
  it('shows startup progress and launch errors', async () => {
    localStorage.setItem(androidPreferenceKey('/project', '/sdk'), 'avd:Tablet');
    const api = bridge();
    api.startEmulator.mockRejectedValue(new Error('Emulator process failed'));
    render(<Harness />);
    fireEvent.click(await screen.findByRole('button', { name: 'Start Emulator' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Emulator process failed'));
    const value = snapshot();
    value.targets[2] = { ...value.targets[2], state: 'starting', canStart: false };
    api.listTargets.mockResolvedValue(value);
    fireEvent.click(screen.getByRole('button', { name: 'Refresh Android targets' }));
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Starting emulator'));
  });
  it('ignores late project/platform results and preserves desktop controls', async () => {
    const api = bridge();
    let resolve!: (value: AndroidTargetSnapshot) => void;
    api.listTargets.mockReturnValueOnce(
      new Promise<AndroidTargetSnapshot>((done) => {
        resolve = done;
      }),
    );
    const view = render(<Harness />);
    view.rerender(<Harness platform="windows" project="/other" />);
    resolve(snapshot());
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: 'Refresh Android targets' })).not.toBeInTheDocument(),
    );
    expect(screen.getByRole('button', { name: 'Build' })).toBeEnabled();
    expect(localStorage.getItem(androidPreferenceKey('/project', '/sdk'))).toBeNull();
    view.rerender(<Harness project="/other" />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('phone'));
  });
  it('scopes preferences to both project and SDK', async () => {
    localStorage.setItem(androidPreferenceKey('/project', '/sdk'), 'avd:Tablet');
    const api = bridge();
    const view = render(<Harness />);
    await screen.findByRole('button', { name: 'Start Emulator' });
    view.rerender(<Harness project="/other" />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Target device' })).toHaveTextContent('phone'));
    api.listTargets.mockResolvedValue({ ...snapshot(), sdkRoot: '/other-sdk' });
    fireEvent.click(screen.getByRole('button', { name: 'Refresh Android targets' }));
    await waitFor(() =>
      expect(localStorage.getItem(androidPreferenceKey('/other', '/other-sdk'))).toBe('device:phone'),
    );
    expect(localStorage.getItem(androidPreferenceKey('/project', '/sdk'))).toBe('avd:Tablet');
  });
});
