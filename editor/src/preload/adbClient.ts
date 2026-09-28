import { execFile } from 'node:child_process';

import type { AndroidDevice, AndroidDeviceConnectionState } from '../common/androidDeviceTypes';

export type AdbCommandResult = {
  stdout: string;
  stderr: string;
};

export type AdbCommandExecutor = (
  executable: string,
  arguments_: string[],
  timeoutMs: number,
) => Promise<AdbCommandResult>;

const executeAdbCommand: AdbCommandExecutor = (executable, arguments_, timeoutMs) =>
  new Promise((resolve, reject) => {
    execFile(
      executable,
      arguments_,
      {
        encoding: 'utf8',
        windowsHide: true,
        timeout: timeoutMs,
        maxBuffer: 1024 * 1024,
      },
      (error, stdout, stderr) => {
        if (error) {
          const detail = String(stderr || stdout || error.message).trim();
          reject(new Error(detail || `ADB command failed: ${error.message}`));
          return;
        }
        resolve({ stdout: String(stdout), stderr: String(stderr) });
      },
    );
  });

const normalizeConnectionState = (value: string): AndroidDeviceConnectionState => {
  if (value === 'device') return 'online';
  if (value === 'offline') return 'offline';
  if (value === 'unauthorized') return 'unauthorized';
  if (value === 'no permissions') return 'no-permissions';
  return 'other';
};

const readableAdbValue = (value: string): string => value.replaceAll('_', ' ').trim();

type ParsedDevice = Omit<AndroidDevice, 'apiLevel'>;

export const parseAdbDevices = (output: string): ParsedDevice[] => {
  const devices: ParsedDevice[] = [];
  for (const sourceLine of output.split(/\r?\n/)) {
    const line = sourceLine.trim();
    if (!line || line.startsWith('List of devices attached') || line.startsWith('* daemon')) continue;

    const serialMatch = line.match(/^(\S+)\s+(.+)$/);
    if (!serialMatch) continue;
    const serial = serialMatch[1];
    let remainder = serialMatch[2].trim();

    let stateDetail = remainder.split(/\s+/, 1)[0] ?? '';
    if (remainder.startsWith('no permissions')) stateDetail = 'no permissions';
    const metadataStart = remainder.search(/(?:^|\s)(?:product|model|device|transport_id):/);
    if (metadataStart >= 0) remainder = remainder.slice(metadataStart).trim();
    else remainder = '';

    const metadata = new Map<string, string>();
    for (const token of remainder.split(/\s+/)) {
      const separator = token.indexOf(':');
      if (separator <= 0) continue;
      metadata.set(token.slice(0, separator), token.slice(separator + 1));
    }

    devices.push({
      serial,
      state: normalizeConnectionState(stateDetail),
      stateDetail,
      model: readableAdbValue(metadata.get('model') ?? ''),
      product: readableAdbValue(metadata.get('product') ?? ''),
      device: readableAdbValue(metadata.get('device') ?? ''),
      transportId: metadata.get('transport_id') ?? '',
      emulator: serial.startsWith('emulator-'),
    });
  }
  return devices;
};

export class AdbClient {
  private versionCache: string | null = null;

  constructor(
    readonly executable: string,
    private readonly executor: AdbCommandExecutor = executeAdbCommand,
  ) {}

  async version(): Promise<string> {
    if (this.versionCache) return this.versionCache;
    const { stdout } = await this.execute(['version']);
    const lines = stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    this.versionCache = lines.find((line) => line.startsWith('Version '))?.slice('Version '.length) ?? lines[0] ?? '';
    return this.versionCache;
  }

  async listDevices(): Promise<AndroidDevice[]> {
    const { stdout } = await this.execute(['devices', '-l']);
    const devices = parseAdbDevices(stdout);
    return Promise.all(
      devices.map(async (device): Promise<AndroidDevice> => {
        if (device.state !== 'online') return device;
        const [apiLevel, model] = await Promise.all([
          this.readDeviceProperty(device.serial, 'ro.build.version.sdk'),
          device.model ? Promise.resolve(device.model) : this.readDeviceProperty(device.serial, 'ro.product.model'),
        ]);
        const parsedApi = Number.parseInt(apiLevel, 10);
        return {
          ...device,
          model: readableAdbValue(model) || device.serial,
          ...(Number.isFinite(parsedApi) ? { apiLevel: parsedApi } : {}),
        };
      }),
    );
  }

  execute(arguments_: string[], timeoutMs = 5000): Promise<AdbCommandResult> {
    return this.executor(this.executable, arguments_, timeoutMs);
  }

  // Install/launch tooling can build on this without exposing arbitrary ADB execution to the renderer.
  executeForDevice(serial: string, arguments_: string[], timeoutMs = 5000): Promise<AdbCommandResult> {
    return this.execute(['-s', serial, ...arguments_], timeoutMs);
  }

  private async readDeviceProperty(serial: string, property: string): Promise<string> {
    try {
      const { stdout } = await this.executeForDevice(serial, ['shell', 'getprop', property], 3000);
      return stdout.trim();
    } catch {
      return '';
    }
  }
}
