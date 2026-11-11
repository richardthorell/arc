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
