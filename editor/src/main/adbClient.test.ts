import { describe, expect, it, vi } from 'vitest';

import {
  AdbClient,
  parseAdbDevices,
  adbNetworkAddress,
  parseAdbMdnsAddresses,
  type AdbCommandExecutor,
} from './adbClient';

describe('AdbClient', () => {
  it('extracts IPv4 and IPv6 endpoints without treating hostnames or USB serials as IPs', () => {
    expect(adbNetworkAddress('192.168.1.20:5555')).toBe('192.168.1.20');
    expect(adbNetworkAddress('[2001:db8::1]:5555')).toBe('2001:db8::1');
    expect(adbNetworkAddress('SERIAL')).toBeUndefined();
    expect(adbNetworkAddress('phone.local:5555')).toBeUndefined();
  });
  it('resolves only exact wireless connect services, not pairing services', async () => {
    const services =
      'List of discovered mdns services\nadb-phone _adb-tls-connect._tcp. 192.168.1.20:37001\nadb-other _adb-tls-pairing._tcp. 192.168.1.21:37002\n';
    expect([...parseAdbMdnsAddresses(services)]).toEqual([['adb-phone._adb-tls-connect._tcp', '192.168.1.20']]);
    const executor: AdbCommandExecutor = async (_file, args) => ({
      stdout:
        args[0] === 'devices'
          ? 'adb-phone._adb-tls-connect._tcp device model:Pixel\n'
          : args[0] === 'mdns'
            ? services
            : '',
      stderr: '',
    });
    expect((await new AdbClient('adb', executor).listDevices())[0]).toMatchObject({
      connectionType: 'network',
      networkAddress: '192.168.1.20',
    });
  });
  it('distinguishes reported USB, network, emulator, and unknown transports', () => {
    const devices = parseAdbDevices(
      'usb-phone device usb:1-2 model:Pixel\n192.168.1.2:5555 device\nadb-phone._adb-tls-connect._tcp device\nemulator-5554 device\nunknown unauthorized\n',
    );
    expect(devices.map((device) => device.connectionType)).toEqual([
      'usb',
      'network',
      'network',
      'emulator',
      'unknown',
    ]);
  });
  it('parses connected, unauthorized, offline, and permission-denied devices', () => {
    const devices = parseAdbDevices(
      `List of devices attached\nR58M123\tdevice product:panther model:Pixel_7 device:panther transport_id:1\nemulator-5554 offline transport_id:2\nABC unauthorized transport_id:3\nXYZ no permissions (user in plugdev group)\n`,
    );

    expect(devices).toEqual([
      expect.objectContaining({ serial: 'R58M123', state: 'online', model: 'Pixel 7', product: 'panther' }),
      expect.objectContaining({ serial: 'emulator-5554', state: 'offline', emulator: true }),
      expect.objectContaining({ serial: 'ABC', state: 'unauthorized' }),
      expect.objectContaining({ serial: 'XYZ', state: 'no-permissions' }),
    ]);
  });

  it('enriches online devices with API level and model while preserving unavailable devices', async () => {
    const executor = vi.fn<AdbCommandExecutor>(async (_executable, arguments_) => {
      const command = arguments_.join(' ');
      if (command === 'devices -l') {
        return {
          stdout:
            'List of devices attached\nSERIAL1 device product:husky model:Pixel_8_Pro device:husky transport_id:1\nSERIAL2 unauthorized transport_id:2\nSERIAL3 device transport_id:3\n',
          stderr: '',
        };
      }
      if (command === '-s SERIAL1 shell getprop ro.build.version.sdk') return { stdout: '35\n', stderr: '' };
      if (command === '-s SERIAL1 shell getprop ro.build.version.release') return { stdout: '15\n', stderr: '' };
      if (command === '-s SERIAL1 shell getprop ro.product.manufacturer') return { stdout: 'Google\n', stderr: '' };
      if (command === '-s SERIAL1 get-devpath') return { stdout: 'usb:1-2\n', stderr: '' };
      if (command === '-s SERIAL3 shell getprop ro.build.version.sdk') return { stdout: '36\n', stderr: '' };
      if (command === '-s SERIAL3 shell getprop ro.product.model') return { stdout: 'Pixel 9 Pro\n', stderr: '' };
      throw new Error(`Unexpected command: ${command}`);
    });

    const devices = await new AdbClient('/sdk/platform-tools/adb', executor).listDevices();

    expect(devices).toEqual([
      expect.objectContaining({
        serial: 'SERIAL1',
        model: 'Pixel 8 Pro',
        apiLevel: 35,
        androidVersion: '15',
        manufacturer: 'Google',
        connectionType: 'usb',
        state: 'online',
      }),
      expect.objectContaining({ serial: 'SERIAL2', state: 'unauthorized' }),
      expect.objectContaining({ serial: 'SERIAL3', model: 'Pixel 9 Pro', apiLevel: 36, state: 'online' }),
    ]);
    expect(devices[1]).not.toHaveProperty('apiLevel');
  });

  it('keeps arbitrary ADB execution internal but provides a device-scoped primitive for later deploy tooling', async () => {
    const executor = vi.fn<AdbCommandExecutor>(async () => ({ stdout: 'ok', stderr: '' }));
    const client = new AdbClient('adb', executor);

    await client.executeForDevice('SERIAL', ['shell', 'echo', 'hello']);

    expect(executor).toHaveBeenCalledWith('adb', ['-s', 'SERIAL', 'shell', 'echo', 'hello'], 5000);
  });
});
