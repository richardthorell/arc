import { execFile } from 'node:child_process';
import { isIP } from 'node:net';

import type { AndroidConnectionType, AndroidDevice, AndroidDeviceConnectionState } from '../common/androidDeviceTypes';

export type AdbCommandResult = {
  stdout: string;
  stderr: string;
};

export type AdbCommandExecutor = (
  executable: string,
  arguments_: string[],
  timeoutMs: number,
) => Promise<AdbCommandResult>;

export const executeAdbCommand: AdbCommandExecutor = (executable, arguments_, timeoutMs) =>
  new Promise((resolve, reject) => {
    execFile(
      executable,
      arguments_,
      {
        encoding: 'utf8',
        shell: false,
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
export const adbNetworkAddress = (endpoint: string): string | undefined => {
  const value = endpoint.replace(/^tcp:/, '');
  const match = value.match(/^\[([^\]]+)\]:\d+$/) ?? value.match(/^(.+):\d+$/);
  return match && isIP(match[1]) ? match[1] : undefined;
};

export const parseAdbMdnsAddresses = (output: string): Map<string, string> => {
  const addresses = new Map<string, string>();
  for (const line of output.split(/\r?\n/)) {
    const [name, service, endpoint] = line.trim().split(/\s+/);
    if (!name || !service || !endpoint || service.replace(/\.$/, '') !== '_adb-tls-connect._tcp') continue;
    const address = adbNetworkAddress(endpoint);
    if (address) addresses.set(`${name}.${service}`.replace(/\.$/, ''), address);
  }
  return addresses;
};
const connectionType = (serial: string, devicePath = ''): AndroidConnectionType => {
  if (serial.startsWith('emulator-')) return 'emulator';
  if (devicePath.startsWith('usb:')) return 'usb';
  if (devicePath.startsWith('tcp:') || /:\d+$/.test(serial) || serial.includes('._tcp')) return 'network';
  return 'unknown';
};

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
    const metadataStart = remainder.search(/(?:^|\s)(?:usb|product|model|device|transport_id):/);
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
      connectionType: connectionType(serial, metadata.has('usb') ? `usb:${metadata.get('usb')}` : ''),
      networkAddress: adbNetworkAddress(serial),
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
    // Wireless debugging may use an mDNS serial; resolve that exact connect service.
    const networkAddresses = devices.some((device) => device.connectionType === 'network' && !device.networkAddress)
      ? await this.execute(['mdns', 'services'], 3000)
          .then((result) => parseAdbMdnsAddresses(result.stdout))
          .catch(() => new Map<string, string>())
      : new Map<string, string>();
    const result: AndroidDevice[] = [];
    for (let offset = 0; offset < devices.length; offset += 4) {
      result.push(
        ...(await Promise.all(
          devices.slice(offset, offset + 4).map(async (device): Promise<AndroidDevice> => {
            const networkAddress = device.networkAddress ?? networkAddresses.get(device.serial.replace(/\.$/, ''));
            if (device.state !== 'online') return { ...device, networkAddress };
            const [apiLevel, model, androidVersion, manufacturer, devicePath] = await Promise.all([
              this.readDeviceProperty(device.serial, 'ro.build.version.sdk'),
              device.model ? Promise.resolve(device.model) : this.readDeviceProperty(device.serial, 'ro.product.model'),
              this.readDeviceProperty(device.serial, 'ro.build.version.release'),
              this.readDeviceProperty(device.serial, 'ro.product.manufacturer'),
              device.connectionType === 'unknown'
                ? this.executeForDevice(device.serial, ['get-devpath'], 3000)
                    .then((result) => result.stdout.trim())
                    .catch(() => '')
                : Promise.resolve(''),
            ]);
            const parsedApi = Number.parseInt(apiLevel, 10);
            return {
              ...device,
              model: readableAdbValue(model) || device.serial,
              androidVersion: androidVersion || undefined,
              manufacturer: manufacturer || undefined,
              networkAddress: networkAddress ?? adbNetworkAddress(devicePath),
              connectionType:
                device.connectionType === 'unknown' ? connectionType(device.serial, devicePath) : device.connectionType,
              ...(Number.isFinite(parsedApi) ? { apiLevel: parsedApi } : {}),
            };
          }),
        )),
      );
    }
    return result;
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
