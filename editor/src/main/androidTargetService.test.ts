import path from 'node:path';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import { AndroidTargetService } from './androidTargetService';

const services: AndroidTargetService[] = [];
afterEach(() => {
  services.forEach((service) => service.dispose());
  services.length = 0;
  vi.useRealTimers();
});
function fixture() {
  const state = {
    root: path.resolve('/sdk'),
    devices: 'phone device model:Pixel_9\n',
    avds: 'Pixel_API_36\n',
    booted: false,
    avdFailure: false,
    adbFailure: false,
    missing: false,
  };
  const execute = vi.fn(async (_file: string, args: string[]) => {
    if (args[0] === 'version') return { stdout: 'Version 36', stderr: '' };
    if (args[0] === 'devices') {
      if (state.adbFailure) throw new Error('ADB server unavailable');
      return { stdout: `List of devices attached\n${state.devices}`, stderr: '' };
    }
    if (args[0] === '-list-avds') {
      if (state.avdFailure) throw new Error('AVD enumeration failed');
      return { stdout: state.avds, stderr: '' };
    }
    if (args.includes('emu')) return { stdout: 'Pixel_API_36\nOK\n', stderr: '' };
    if (args.includes('sys.boot_completed')) return { stdout: state.booted ? '1\n' : '0\n', stderr: '' };
    return { stdout: '36\n', stderr: '' };
  });
  const launch = vi.fn<(file: string, args: string[], exited: (error: string) => void) => void>();
  const service = new AndroidTargetService(
    () => ({ values: { 'platform.android.sdkPath': state.root } }) as unknown as EditorSettingsSnapshot,
    execute,
    launch,
    () => !state.missing,
  );
  services.push(service);
  return { state, execute, launch, service, target: { sdkRoot: state.root, id: 'avd:Pixel_API_36' } };
}
describe('AndroidTargetService', () => {
  it('combines physical devices and stopped AVDs, retaining unavailable connections', async () => {
    const { state, service } = fixture();
    state.devices += 'second unauthorized model:Pixel_9\nthird offline model:Pixel_9\n';
    const snapshot = await service.listTargets();
    expect(snapshot.targets).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'device:phone', ready: true, apiLevel: 36 }),
        expect.objectContaining({ id: 'device:second', state: 'unauthorized', ready: false }),
        expect.objectContaining({ id: 'device:third', state: 'offline', ready: false }),
        expect.objectContaining({ id: 'avd:Pixel_API_36', state: 'stopped', canStart: true }),
      ]),
    );
  });
  it('does not hide physical devices when AVD discovery fails', async () => {
    const { state, service } = fixture();
    state.avdFailure = true;
    const snapshot = await service.listTargets();
    expect(snapshot.targets[0].ready).toBe(true);
    expect(snapshot.diagnostics.join()).toContain('AVD enumeration failed');
  });
  it('reports missing tools and retains stopped AVDs when ADB fails', async () => {
    const { state, service } = fixture();
    state.missing = true;
    expect((await service.listTargets()).error).toContain('ADB unavailable');
    state.missing = false;
    state.adbFailure = true;
    expect((await service.listTargets()).targets).toContainEqual(
      expect.objectContaining({ state: 'stopped', canStart: false }),
    );
  });
  it('merges running AVDs and requires Android boot completion', async () => {
    const { state, service, target } = fixture();
    state.devices = 'emulator-5554 device model:Pixel\n';
    expect((await service.listTargets()).targets).toHaveLength(1);
    await expect(service.resolveTarget(target)).rejects.toThrow('not booted');
    state.booted = true;
    expect(await service.resolveTarget(target)).toEqual({ sdkRoot: state.root, serial: 'emulator-5554' });
    state.devices = 'other device model:Pixel\n';
    await expect(service.resolveTarget(target)).rejects.toThrow('unavailable');
  });
  it('keeps multiple instances distinct and rejects an ambiguous AVD identity', async () => {
    const { state, service, target } = fixture();
    state.booted = true;
    state.devices = 'emulator-5554 device\nemulator-5556 device\n';
    expect((await service.listTargets()).targets.map((entry) => entry.id)).toEqual([
      'device:emulator-5554',
      'device:emulator-5556',
    ]);
    await expect(service.resolveTarget(target)).rejects.toThrow('ambiguous');
    await expect(service.startEmulator(target)).rejects.toThrow('ambiguous');
  });
  it('exposes an unrecognized emulator by serial', async () => {
    const { state, service, execute } = fixture();
    state.devices = 'emulator-5554 device\n';
    state.booted = true;
    const original = execute.getMockImplementation()!;
    execute.mockImplementation(async (file, args) => {
      if (args.includes('emu')) throw new Error('No console');
      return original(file, args);
    });
    expect((await service.listTargets()).targets).toContainEqual(
      expect.objectContaining({ id: 'device:emulator-5554', ready: true }),
    );
  });
  it('deduplicates startup and reuses already running emulators', async () => {
    vi.useFakeTimers();
    const { service, target, launch, state } = fixture();
    await Promise.all([service.startEmulator(target), service.startEmulator(target)]);
    await service.startEmulator(target);
    expect(launch).toHaveBeenCalledTimes(1);
    expect(launch.mock.calls[0].slice(0, 2)).toEqual([
      path.join(state.root, 'emulator', process.platform === 'win32' ? 'emulator.exe' : 'emulator'),
      ['-avd', 'Pixel_API_36'],
    ]);
    expect((await service.listTargets()).targets.find((entry) => entry.id === target.id)?.state).toBe('starting');
    state.devices = 'emulator-5554 device\n';
    state.booted = true;
    await vi.advanceTimersByTimeAsync(2000);
    expect((await service.resolveTarget(target)).serial).toBe('emulator-5554');
    await service.startEmulator(target);
    expect(launch).toHaveBeenCalledTimes(1);
  });
  it('rejects renderer-supplied AVD names that were not discovered', async () => {
    const { service, target, launch } = fixture();
    await expect(service.startEmulator({ ...target, id: 'avd:arbitrary --args' })).rejects.toThrow('unavailable');
    expect(launch).not.toHaveBeenCalled();
    await expect(service.startEmulator({ ...target, id: 'device:phone' })).rejects.toThrow('physical device');
  });
  it('reports process failure and permits retry without killing any process', async () => {
    const { service, target, launch } = fixture();
    await service.startEmulator(target);
    launch.mock.calls[0][2]('Process exited: 1');
    expect((await service.listTargets()).targets.find((entry) => entry.id === target.id)).toMatchObject({
      state: 'failed',
      diagnostic: 'Process exited: 1',
      canStart: true,
    });
    await service.startEmulator(target);
    expect(launch).toHaveBeenCalledTimes(2);
  });
  it('reports synchronous spawn failure', async () => {
    const { service, target, launch } = fixture();
    launch.mockImplementation(() => {
      throw new Error('ENOENT');
    });
    await expect(service.startEmulator(target)).rejects.toThrow('ENOENT');
    expect((await service.listTargets()).targets.find((entry) => entry.id === target.id)?.diagnostic).toBe('ENOENT');
  });
  it('times out waiting after three minutes but accepts later boot completion', async () => {
    vi.useFakeTimers();
    const { service, target, state } = fixture();
    await service.startEmulator(target);
    await vi.advanceTimersByTimeAsync(180_000);
    expect((await service.listTargets()).targets.find((entry) => entry.id === target.id)?.diagnostic).toContain(
      'was not stopped',
    );
    state.devices = 'emulator-5554 device\n';
    state.booted = true;
    expect((await service.resolveTarget(target)).serial).toBe('emulator-5554');
  });
  it('rejects stale SDK references and late SDK discovery', async () => {
    const { service, state, target, launch } = fixture();
    const pending = service.listTargets();
    state.root = path.resolve('/other-sdk');
    await expect(pending).rejects.toThrow('SDK changed');
    expect(() => service.startEmulator(target)).toThrow('stale');
    await expect(service.resolveTarget(target)).rejects.toThrow('stale');
    expect(launch).not.toHaveBeenCalled();
  });
});
