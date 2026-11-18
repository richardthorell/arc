import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { BuildService } from './buildService';
import * as msbuild from './msbuildCompileTasks';

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

  it('passes the selected configuration to CMake configure instead of using the CLI default', async () => {
    const { service, process: child } = setup();
    const pending = service.execute({ action: 'configure', configuration: 'Debug' });
    expect(spawn).toHaveBeenCalledWith(
      process.execPath,
      ['configure', '--project', activeProject.descriptorPath, '--config', 'Debug'],
      expect.objectContaining({ cwd: activeProject.projectRoot }),
    );
    child.emit('close', 0);
    await expect(pending).resolves.toMatchObject({ state: 'succeeded', configuration: 'Debug' });
  });

  it('emits separate, untruncated stdout/stderr chunks to the history observer', async () => {
    const child = new FakeBuildProcess();
    vi.mocked(spawn).mockReturnValue(child as unknown as ChildProcessWithoutNullStreams);
    const raw = vi.fn();
    const service = new BuildService(
      () => activeProject,
      () => process.execPath,
      { command: vi.fn() },
      vi.fn(),
      undefined,
      undefined,
      () => false,
      raw,
    );
    const pending = service.execute({ action: 'configure' });
    const large = 'x'.repeat(256 * 1024);
    child.stdout.emit('data', Buffer.from(large));
    child.stderr.emit('data', Buffer.from('failure detail'));
    child.emit('close', 0);
    await pending;
    expect(raw).toHaveBeenCalledWith(activeProject, expect.any(String), 'stdout', large);
    expect(raw).toHaveBeenCalledWith(activeProject, expect.any(String), 'stderr', 'failure detail');
  });

  it('publishes the editor module graph during a first normal Build, then preserves it after reload', async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-bm6-build-'));
    try {
      const project = {
        ...activeProject,
        projectRoot: root,
        descriptorPath: path.join(root, 'Game.arcproject'),
        descriptor: {
          ...activeProject.descriptor,
          guid: 'game-bm6',
          paths: { source: 'Source', saved: 'Saved', build: 'Build' },
          modules: [{ id: 'game.editor', kind: 'editor', target: 'MyEditor', enabled: true }],
        },
      } as ArcProjectCandidate;
      const child = new FakeBuildProcess();
      vi.mocked(spawn).mockReturnValue(child as unknown as ChildProcessWithoutNullStreams);
      const publish = vi.fn();
      const host = { command: vi.fn().mockResolvedValue({ succeeded: true }) };
      const service = new BuildService(
        () => project,
        () => process.execPath,
        host,
        publish,
      );
      const pending = service.execute({ action: 'build', configuration: 'Debug' });
      const buildDir = path.join(root, 'Build', 'default');
      expect(fs.existsSync(path.join(buildDir, '.cmake', 'api', 'v1', 'query', 'codemodel-v2'))).toBe(true);

      const replyDir = path.join(buildDir, '.cmake', 'api', 'v1', 'reply');
      fs.mkdirSync(replyDir, { recursive: true });
      fs.writeFileSync(
        path.join(replyDir, 'index-test.json'),
        JSON.stringify({
          cmake: { generator: { name: 'Ninja' } },
          reply: { 'codemodel-v2': { jsonFile: 'codemodel.json' } },
        }),
      );
      fs.writeFileSync(
        path.join(replyDir, 'codemodel.json'),
        JSON.stringify({
          configurations: [{ name: '', targets: [{ id: 'editor', name: 'MyEditor', jsonFile: 'target-editor.json' }] }],
        }),
      );
      fs.writeFileSync(
        path.join(replyDir, 'target-editor.json'),
        JSON.stringify({
          id: 'editor',
          name: 'MyEditor',
          type: 'MODULE_LIBRARY',
          artifacts: [{ path: 'MyEditor.dll' }],
        }),
      );
      const output = path.join(buildDir, 'MyEditor.dll');
      child.stdout.emit('data', Buffer.from(`MyEditor.vcxproj -> ${output}\n`));

      const saved = path.join(root, 'Saved', 'Editor');
      fs.mkdirSync(saved, { recursive: true });
      fs.writeFileSync(
        path.join(saved, 'active-build.json'),
        JSON.stringify({
          moduleManifest: 'Build/default/arc-modules-Debug.json',
        }),
      );
      fs.writeFileSync(
        path.join(buildDir, 'arc-modules-Debug.json'),
        JSON.stringify({
          modules: [{ id: 'game.editor', path: output }],
        }),
      );

      // Ninja creates its log during the native compilation invocation.
      fs.writeFileSync(
        path.join(buildDir, '.ninja_log'),
        '# ninja log v5\n10\t80\t123\tCMakeFiles/MyEditor.dir/main.cpp.obj\tcommand-hash\n',
      );
      fs.writeFileSync(
        path.join(buildDir, 'compile_commands.json'),
        JSON.stringify([
          {
            directory: buildDir,
            file: path.join(root, 'Source', 'main.cpp'),
            command: 'c++ -c Source/main.cpp -o CMakeFiles/MyEditor.dir/main.cpp.obj',
          },
        ]),
      );
      child.emit('close', 0);
      await expect(pending).resolves.toMatchObject({ state: 'succeeded' });
      expect(service.snapshot().targetGraph).toMatchObject({
        configuration: 'Debug',
        rootTargetId: 'editor',
        revision: 2,
        nodes: [{ name: 'MyEditor', artifactObservedAt: expect.any(String) }],
      });
      expect(service.snapshot().targetGraph?.nodes[0]).not.toHaveProperty('satisfiedAt');
      expect(service.snapshot().compileTrace).toMatchObject({
        source: 'ninja-log',
        clock: 'ninja-invocation',
        tasks: [
          {
            output: 'CMakeFiles/MyEditor.dir/main.cpp.obj',
            sourceFile: 'Source/main.cpp',
            targetName: 'MyEditor',
            durationMs: 70,
          },
        ],
      });
      expect(publish.mock.calls.some(([entry]) => entry.targetGraph?.revision === 1)).toBe(true);
      expect(host.command).toHaveBeenCalledOnce();
    } finally {
      fs.rmSync(root, { recursive: true, force: true });
    }
  });

  it('forwards native MSBuild telemetry flags without changing failed command reporting', async () => {
    const capture = vi.spyOn(msbuild, 'beginMsbuildCompileTrace').mockReturnValue({
      args: ['--msbuild-timing-log', '/project/.arc-monitor/trace.log'],
      finish: () => ({
        source: 'msbuild-file-log',
        clock: 'msbuild-cl-events',
        truncated: false,
        tasks: [{ output: 'CL task #7', startMs: 0, endMs: 200, durationMs: 200 }],
      }),
    });
    try {
      const { service, process: child } = setup();
      const pending = service.execute({ action: 'build', configuration: 'Debug' });
      expect(spawn).toHaveBeenCalledWith(
        process.execPath,
        [
          'build',
          '--project',
          activeProject.descriptorPath,
          '--config',
          'Debug',
          '--msbuild-timing-log',
          '/project/.arc-monitor/trace.log',
        ],
        expect.objectContaining({ cwd: activeProject.projectRoot }),
      );
      child.emit('close', 1);
      await expect(pending).rejects.toThrow('code 1');
      expect(service.snapshot().compileTrace).toEqual({
        source: 'msbuild-file-log',
        clock: 'msbuild-cl-events',
        truncated: false,
        tasks: [{ output: 'CL task #7', startMs: 0, endMs: 200, durationMs: 200 }],
      });
    } finally {
      capture.mockRestore();
    }
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
