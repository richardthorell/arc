import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

import type { EditorSettingsSnapshot } from '../common/editorWorkflowTypes';
import type { AndroidTarget, AndroidTargetReference, AndroidTargetSnapshot } from '../common/androidDeviceTypes';
import { AdbClient, executeAdbCommand, type AdbCommandExecutor } from './adbClient';

const executable = (name: string) => (process.platform === 'win32' ? `${name}.exe` : name);
export const resolveAndroidSdkRoot = (settings: EditorSettingsSnapshot | null | undefined): string => {
  const validation = settings?.pathValidation?.['platform.android.sdkPath'];
  const root =
    validation?.valid && validation.resolvedPath
      ? validation.resolvedPath
      : settings?.values?.['platform.android.sdkPath'];
  return typeof root === 'string' && root.trim() ? path.resolve(root.trim()) : '';
};
export const resolveAndroidAdbPath = (settings: EditorSettingsSnapshot | null | undefined): string => {
  const root = resolveAndroidSdkRoot(settings);
  return root ? path.join(root, 'platform-tools', executable('adb')) : '';
};
type Launch = (file: string, args: string[], onExit: (error: string) => void) => void;
const launchEmulator: Launch = (file, args, onExit) => {
  const child = spawn(file, args, { shell: false, windowsHide: true, detached: true, stdio: 'ignore' });
  child.once('error', (error) => onExit(error.message));
  child.once('exit', (code) => onExit(`Emulator exited before becoming ready (code ${code ?? 'unknown'})`));
  child.unref();
};
type Startup = {
  sdkRoot: string;
  avdName: string;
  deadline: number;
  error?: string;
  timer?: ReturnType<typeof setTimeout>;
  timeout?: ReturnType<typeof setTimeout>;
};
const message = (error: unknown) => (error instanceof Error ? error.message : String(error));
const avdId = (name: string) => `avd:${encodeURIComponent(name)}`;

/** Owns discovery/startup only. APK installation and application launch belong to the deploy service. */
export class AndroidTargetService {
  private readonly startups = new Map<string, Startup>();
  private readonly starts = new Map<string, Promise<void>>();
  private readonly scans = new Map<string, Promise<AndroidTargetSnapshot>>();
  private disposed = false;
  constructor(
    private readonly settings: () => EditorSettingsSnapshot | null,
    private readonly execute: AdbCommandExecutor = executeAdbCommand,
    private readonly launch: Launch = launchEmulator,
    private readonly exists: (file: string) => boolean = fs.existsSync,
    private readonly now: () => number = Date.now,
  ) {}

  async listTargets(): Promise<AndroidTargetSnapshot> {
    const root = resolveAndroidSdkRoot(this.settings());
    let scan = this.scans.get(root);
    if (!scan) {
      scan = this.discover(root).finally(() => {
        if (this.scans.get(root) === scan) this.scans.delete(root);
      });
      this.scans.set(root, scan);
    }
    const result = await scan;
    if (root !== resolveAndroidSdkRoot(this.settings()))
      throw new Error('Android SDK changed during discovery; refresh targets');
    return result;
  }

  private async discover(sdkRoot: string): Promise<AndroidTargetSnapshot> {
    const adbPath = sdkRoot ? path.join(sdkRoot, 'platform-tools', executable('adb')) : '';
    const emulatorPath = sdkRoot ? path.join(sdkRoot, 'emulator', executable('emulator')) : '';
    const result: AndroidTargetSnapshot = {
      sdkRoot,
      adbPath,
      adbVersion: '',
      available: false,
      devices: [],
      targets: [],
      emulatorAvailable: !!emulatorPath && this.exists(emulatorPath),
      diagnostics: [],
      error: '',
      refreshedAt: new Date(this.now()).toISOString(),
    };
    const client = new AdbClient(adbPath, this.execute);
    const avds = new Set<string>();
    await Promise.all([
      (async () => {
        if (!adbPath || !this.exists(adbPath)) {
          result.error = 'ADB unavailable. Configure an Android SDK with platform-tools in Preferences.';
          result.diagnostics.push(result.error);
          return;
        }
        try {
          [result.adbVersion, result.devices] = await Promise.all([client.version(), client.listDevices()]);
          result.available = true;
        } catch (error) {
          result.error = message(error);
          result.diagnostics.push(result.error);
        }
      })(),
      (async () => {
        if (!result.emulatorAvailable) {
          result.diagnostics.push(
            'Android Emulator is not installed in the configured SDK. Install it and create an AVD using Android Studio.',
          );
          return;
        }
        try {
          const { stdout } = await this.execute(emulatorPath, ['-list-avds'], 5000);
          for (const name of stdout
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean))
            avds.add(name);
        } catch (error) {
          result.emulatorAvailable = false;
          result.diagnostics.push(`Emulator discovery failed: ${message(error)}`);
        }
      })(),
    ]);
    // Bound extra emulator queries; one unresponsive device must not hide the others.
    for (let offset = 0; offset < result.devices.length; offset += 4) {
      const targets = await Promise.all(
        result.devices.slice(offset, offset + 4).map(async (device): Promise<AndroidTarget> => {
          const target: AndroidTarget = {
            id: `device:${device.serial}`,
            kind: device.emulator ? 'emulator' : 'device',
            name: device.model || device.serial,
            serial: device.serial,
            apiLevel: device.apiLevel,
            state: device.state,
            ready: device.state === 'online',
            canStart: false,
          };
          if (device.state === 'unauthorized')
            target.diagnostic = 'Authorize USB debugging on this device, then refresh.';
          else if (device.state === 'offline') target.diagnostic = 'Device is offline. Reconnect it and refresh.';
          else if (device.state === 'no-permissions')
            target.diagnostic = 'ADB lacks device permissions. Configure USB access on this host and refresh.';
          if (device.emulator) {
            try {
              const { stdout } = await client.executeForDevice(device.serial, ['emu', 'avd', 'name'], 3000);
              const name = stdout
                .split(/\r?\n/)
                .map((line) => line.trim())
                .find((line) => line && line !== 'OK' && !line.startsWith('KO:'));
              if (name) {
                target.avdName = name;
                target.name = name;
              }
            } catch {
              /* Still expose the emulator by serial. */
            }
            if (device.state === 'online') {
              try {
                const { stdout } = await client.executeForDevice(
                  device.serial,
                  ['shell', 'getprop', 'sys.boot_completed'],
                  3000,
                );
                target.ready = stdout.trim() === '1';
              } catch {
                target.ready = false;
              }
              if (!target.ready) target.state = 'starting';
            }
          }
          return target;
        }),
      );
      result.targets.push(...targets);
    }
    for (const name of avds) {
      const matching = result.targets.filter((target) => target.avdName === name);
      if (matching.length === 1) matching[0].id = avdId(name);
      if (!matching.length)
        result.targets.push({
          id: avdId(name),
          name,
          avdName: name,
          kind: 'emulator',
          state: 'stopped',
          ready: false,
          canStart: result.available && result.emulatorAvailable,
        });
    }
    for (const target of result.targets) {
      const startup = this.startups.get(`${sdkRoot}\0${target.id}`);
      if (!startup) continue;
      if (target.ready) {
        this.clearStartup(`${sdkRoot}\0${target.id}`);
        continue;
      }
      if (startup.error) {
        target.state = 'failed';
        target.diagnostic = startup.error;
      } else {
        target.state = 'starting';
        target.canStart = false;
      }
    }
    result.targets.sort(
      (a, b) => Number(b.ready) - Number(a.ready) || a.name.localeCompare(b.name) || a.id.localeCompare(b.id),
    );
    return result;
  }

  startEmulator(reference: AndroidTargetReference): Promise<void> {
    this.validate(reference);
    const key = `${reference.sdkRoot}\0${reference.id}`;
    const pending = this.starts.get(key);
    if (pending) return pending;
    const work = this.start(reference, key).finally(() => this.starts.delete(key));
    this.starts.set(key, work);
    return work;
  }

  private async start(reference: AndroidTargetReference, key: string): Promise<void> {
    const snapshot = await this.listTargets();
    this.validate(reference);
    const target = snapshot.targets.find((entry) => entry.id === reference.id);
    if (!target) throw new Error('Selected emulator is unavailable or ambiguous. Refresh and select a target.');
    if (target.kind !== 'emulator')
      throw new Error('Start Emulator requires an emulator target, not a physical device.');
    if (target.ready || target.state === 'starting') return;
    if (!target.canStart || !target.avdName)
      throw new Error('Selected target cannot be started. Check Android SDK and device diagnostics.');
    this.clearStartup(key);
    const startup: Startup = { sdkRoot: reference.sdkRoot, avdName: target.avdName, deadline: this.now() + 180_000 };
    this.startups.set(key, startup);
    startup.timeout = setTimeout(() => {
      if (this.startups.get(key) !== startup || startup.error) return;
      startup.error =
        'Emulator startup timed out after three minutes. It was not stopped; check its window and refresh.';
      if (startup.timer) clearTimeout(startup.timer);
    }, 180_000);
    startup.timeout.unref?.();
    try {
      this.launch(
        path.join(reference.sdkRoot, 'emulator', executable('emulator')),
        ['-avd', target.avdName],
        (error) => {
          if (this.startups.get(key) === startup) startup.error = error;
        },
      );
    } catch (error) {
      startup.error = message(error);
      throw error;
    }
    this.scans.delete(reference.sdkRoot);
    this.monitor(key, startup);
  }

  private monitor(key: string, startup: Startup): void {
    startup.timer = setTimeout(() => {
      void (async () => {
        if (this.disposed || this.startups.get(key) !== startup) return;
        if (startup.sdkRoot !== resolveAndroidSdkRoot(this.settings())) {
          this.clearStartup(key);
          return;
        }
        if (this.now() >= startup.deadline)
          startup.error =
            'Emulator startup timed out after three minutes. It was not stopped; check its window and refresh.';
        if (startup.error) return;
        try {
          await this.listTargets();
        } catch {
          /* Discovery reports details; keep the bounded startup wait. */
        }
        if (this.startups.get(key) === startup) this.monitor(key, startup);
      })();
    }, 2000);
    startup.timer.unref?.();
  }

  async resolveTarget(reference: AndroidTargetReference): Promise<{ serial: string; sdkRoot: string }> {
    this.validate(reference);
    const snapshot = await this.listTargets();
    this.validate(reference);
    const target = snapshot.targets.find((entry) => entry.id === reference.id);
    if (!target?.ready || !target.serial)
      throw new Error(
        'Selected Android target is unavailable, ambiguous, or not booted. Refresh and select a ready target.',
      );
    return { serial: target.serial, sdkRoot: snapshot.sdkRoot };
  }
  private validate(reference: AndroidTargetReference): void {
    if (
      this.disposed ||
      !reference ||
      typeof reference.id !== 'string' ||
      !reference.id ||
      reference.id.length > 512 ||
      typeof reference.sdkRoot !== 'string' ||
      !reference.sdkRoot ||
      reference.sdkRoot !== resolveAndroidSdkRoot(this.settings())
    )
      throw new Error('Invalid or stale Android target. Refresh after configuring the SDK.');
  }
  private clearStartup(key: string): void {
    const startup = this.startups.get(key);
    if (startup?.timer) clearTimeout(startup.timer);
    if (startup?.timeout) clearTimeout(startup.timeout);
    this.startups.delete(key);
  }
  dispose(): void {
    this.disposed = true;
    for (const key of this.startups.keys()) this.clearStartup(key);
    // Emulators remain running; the editor does not own their lifetime after launch.
  }
}
