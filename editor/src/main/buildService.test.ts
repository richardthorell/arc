import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { EventEmitter } from 'node:events';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { BuildService } from './buildService';

vi.mock('node:child_process', () => ({ spawn: vi.fn() }));

class FakeBuildProcess extends EventEmitter {
  stdout = new EventEmitter();
  stderr = new EventEmitter();
  kill = vi.fn(() => true);
}

const activeProject = {
  projectRoot: '/fake/project',
  descriptorPath: '/fake/project/Test.arcproject',
  writable: true,
  descriptor: {
    paths: { source: 'Source', saved: 'Saved' },
  },
} as ArcProjectCandidate;

const setup = () => {
  const child = new FakeBuildProcess();
  vi.mocked(spawn).mockReturnValue(child as unknown as ChildProcessWithoutNullStreams);
  const publish = vi.fn();
  const service = new BuildService(
    () => activeProject,
    () => process.execPath,
    { command: vi.fn() },
    publish,
  );
  return { service, process: child, publish };
};

beforeEach(() => {
  vi.mocked(spawn).mockReset();
});

describe('BuildService command history', () => {
  it('records measured successful configure commands and the selected configuration', async () => {
    const { service, process } = setup();
    const request = service.execute({ action: 'configure', configuration: 'Debug' });

    process.stdout.emit('data', Buffer.from('Configuring project\n'));
    process.emit('close', 0);
    await expect(request).resolves.toMatchObject({ state: 'succeeded', configuration: 'Debug' });

    const history = service.snapshot().history ?? [];
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({
      kind: 'configure',
      configuration: 'Debug',
      result: 'succeeded',
      warningCount: 0,
      errorCount: 0,
    });
    expect(history[0].durationMs).toBeGreaterThanOrEqual(0);
    expect(history[0].runId).toEqual(expect.any(String));
    expect(service.snapshot().runId).toBe(history[0].runId);
  });

  it('records compiler errors and failed commands', async () => {
    const { service, process } = setup();
    const request = service.execute({ action: 'configure' });

    process.stderr.emit('data', Buffer.from('Source/main.cpp:9: error: broken\n'));
    process.emit('close', 1);
    await expect(request).rejects.toThrow('code 1');

    expect(service.snapshot().history?.[0]).toMatchObject({
      kind: 'configure',
      result: 'failed',
      errorCount: 1,
    });
  });

  it('preserves CMake configuration diagnostics with error severity on stdout', async () => {
    const { service, process } = setup();
    const request = service.execute({ action: 'build' });

    process.stdout.emit(
      'data',
      Buffer.from(
        'CMake Error at CMakeLists.txt:9 (find_package):\n' +
          '  Could not find a package configuration file provided by "ARC".\n' +
          'arc-project: CMake configuration failed for /project/Build/default; pass --sdk.\n',
      ),
    );
    process.emit('close', 1);
    await expect(request).rejects.toThrow('code 1');

    const snapshot = service.snapshot();
    expect(snapshot.state).toBe('failed');
    expect(snapshot.diagnostics.some((value) => value.severity === 'error' && value.message.includes('CMake'))).toBe(
      true,
    );
    expect(snapshot.diagnostics.some((value) => value.message.includes('Could not find a package'))).toBe(true);
    expect(snapshot.history?.[0]).toMatchObject({ kind: 'build', result: 'failed' });
  });

  it('does not splice partial CMake stdout and CLI stderr into one diagnostic', async () => {
    const { service, process } = setup();
    const request = service.execute({ action: 'build' });

    process.stdout.emit('data', Buffer.from('CMake Error at CMakeLists.txt:9'));
    process.stderr.emit('data', Buffer.from('arc-project: CMake configuration failed\n'));
    process.stdout.emit('data', Buffer.from(' (find_package):\n'));
    process.emit('close', 1);
    await expect(request).rejects.toThrow('code 1');

    const messages = service.snapshot().diagnostics.map((item) => item.message);
    expect(messages.some((message) => message.includes('CMake Error at CMakeLists.txt'))).toBe(true);
    expect(messages.some((message) => message.includes('arc-project: CMake configuration failed'))).toBe(true);
    expect(messages.some((message) => message.includes('arc-project:') && message.includes('find_package'))).toBe(
      false,
    );
  });

  it('shows a useful diagnostic if the tool exits nonzero without any output', async () => {
    const { service, process } = setup();
    const request = service.execute({ action: 'build' });

    process.emit('close', 1);
    await expect(request).rejects.toThrow('code 1');

    const snapshot = service.snapshot();
    expect(snapshot.diagnostics).toEqual([
      expect.objectContaining({
        severity: 'error',
        message: expect.stringContaining('See CMake output and ARC SDK settings'),
      }),
    ]);
    expect(snapshot.history?.[0].errorCount).toBe(1);
  });

  it('preserves cancellation when the child subsequently closes', async () => {
    const { service, process } = setup();
    const request = service.execute({ action: 'configure' });

    const cancelled = await service.execute({ action: 'cancel' });
    expect(cancelled.state).toBe('cancelled');
    expect(process.kill).toHaveBeenCalledOnce();
    process.emit('close', null);

    await expect(request).rejects.toThrow('cancelled');
    expect(service.snapshot().state).toBe('cancelled');
    expect(service.snapshot().history?.[0].result).toBe('cancelled');
  });
});
