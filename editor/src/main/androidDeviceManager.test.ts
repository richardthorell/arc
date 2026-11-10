import { EventEmitter } from 'node:events';
import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('node:fs', () => ({ default: { existsSync: vi.fn() } }));
vi.mock('node:child_process', () => ({ spawn: vi.fn() }));
vi.mock('electron', () => ({ dialog: { showOpenDialog: vi.fn(), showMessageBox: vi.fn(), showErrorBox: vi.fn() } }));
import fs from 'node:fs';
import { spawn } from 'node:child_process';
import { dialog } from 'electron';
import { openAndroidDeviceManager } from './androidDeviceManager';

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(fs.existsSync).mockReturnValue(false);
  vi.mocked(dialog.showOpenDialog).mockResolvedValue({ canceled: false, filePaths: ['/custom/studio'] });
  vi.mocked(dialog.showMessageBox).mockResolvedValue({ response: 0, checkboxChecked: false });
});

describe('Android Device Manager launcher', () => {
  it('deduplicates requests and launches the user-located Studio without a shell', async () => {
    const child = Object.assign(new EventEmitter(), { unref: vi.fn() });
    vi.mocked(spawn).mockImplementation(() => {
      queueMicrotask(() => child.emit('spawn'));
      return child as unknown as ReturnType<typeof spawn>;
    });
    const first = openAndroidDeviceManager();
    expect(openAndroidDeviceManager()).toBe(first);
    await first;
    expect(dialog.showOpenDialog).toHaveBeenCalledOnce();
    expect(spawn).toHaveBeenCalledWith(
      process.platform === 'darwin' ? '/usr/bin/open' : '/custom/studio',
      process.platform === 'darwin' ? ['-a', '/custom/studio'] : [],
      expect.objectContaining({ shell: false, windowsHide: true, detached: true }),
    );
    expect(child.unref).toHaveBeenCalledOnce();
  });
  it('does nothing when locating Studio is canceled', async () => {
    vi.mocked(dialog.showOpenDialog).mockResolvedValue({ canceled: true, filePaths: [] });
    await openAndroidDeviceManager();
    expect(spawn).not.toHaveBeenCalled();
  });
  it('shows launch failures instead of an unhandled rejection', async () => {
    vi.mocked(spawn).mockImplementation(() => {
      throw new Error('Studio unavailable');
    });
    await openAndroidDeviceManager();
    expect(dialog.showErrorBox).toHaveBeenCalledWith('Cannot open Android Device Manager', 'Studio unavailable');
  });
});
