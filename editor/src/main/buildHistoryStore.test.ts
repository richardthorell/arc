import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import type { BuildMonitorRun, BuildMonitorEvent } from '../common/buildMonitorTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import { BuildMonitorService } from './buildMonitorService';
import { BuildHistoryStore } from './buildHistoryStore';

const folders: string[] = [];
const createProject = (guid = 'game-a'): ArcProjectCandidate => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-build-history-'));
  folders.push(root);
  return {
    descriptor: { guid, paths: { saved: 'Saved' } },
    descriptorPath: path.join(root, 'Game.arcproject'),
    projectRoot: root,
    writable: true,
  } as ArcProjectCandidate;
};
const run = (id: string, guid = 'game-a'): BuildMonitorRun => ({
  id,
  projectGuid: guid,
  source: 'command',
  action: 'build',
  configuration: 'Debug',
  startedAt: '2026-11-16T12:00:00.000Z',
  state: 'running',
  stages: [{ id: `${id}:compile`, name: 'Compile', state: 'running' }],
  warningCount: 0,
  errorCount: 0,
});

afterEach(() => {
  for (const root of folders.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('BuildHistoryStore (BM2)', () => {
  it('persists completed build data, stdout/stderr, diagnostics, and stages across editor restarts', () => {
    const project = createProject();
    const storage = new BuildHistoryStore();
    storage.activate(project);
    const saved = run('test-run-1');
    storage.saveRun(saved);
    storage.saveEvent({
      kind: 'run-started',
      sequence: 1,
      timestamp: saved.startedAt,
      runId: saved.id,
      run: saved,
    });
    storage.appendRaw(saved.id, 'stdout', 'compile output\n', saved.startedAt);
    storage.appendRaw(saved.id, 'stderr', 'fatal error: expected output\n', saved.startedAt);
    storage.saveEvent({
      kind: 'diagnostic',
      sequence: 2,
      timestamp: saved.startedAt,
      runId: saved.id,
      stageId: saved.stages[0].id,
      diagnostic: { sequence: 1, severity: 'error', message: 'expected output', file: 'game.cpp', line: 8 },
    });
    saved.state = 'failed';
    saved.error = 'Compilation failed';
    saved.completedAt = '2026-11-16T12:00:02.000Z';
    saved.durationMs = 2000;
    storage.saveRun(saved);

    const reopened = new BuildHistoryStore();
    reopened.activate(project);
    expect(reopened.list()).toMatchObject({ projectGuid: 'game-a', total: 1 });
    expect(reopened.list().runs[0]).toMatchObject({ id: 'test-run-1', state: 'failed', durationMs: 2000 });

    const detail = reopened.details(saved.id);
    expect(detail?.logs.items).toEqual([
      { stream: 'stdout', text: 'compile output\n', timestamp: saved.startedAt },
      { stream: 'stderr', text: 'fatal error: expected output\n', timestamp: saved.startedAt },
    ]);
    expect(detail?.events.items.map((event) => event.kind)).toEqual(['run-started', 'diagnostic']);
    expect(detail?.events.items[1]).toMatchObject({ stageId: 'test-run-1:compile' });
  });

  it('pages raw output by byte cursor without truncating large logs or duplicating chunks', () => {
    const project = createProject();
    const store = new BuildHistoryStore();
    store.activate(project);
    const saved = run('big-log');
    store.saveRun(saved);
    const original = 'a'.repeat(900_000) + '🌿' + 'tail';
    store.appendRaw(saved.id, 'stdout', original, saved.startedAt);
    let cursor = 0;
    let combined = '';
    let reads = 0;
    do {
      const page = store.details(saved.id, cursor);
      expect(page).not.toBeNull();
      combined += page!.logs.items.map((item) => item.text).join('');
      if (!page!.logs.hasMore) break;
      expect(page!.logs.nextCursor).toBeGreaterThan(cursor);
      cursor = page!.logs.nextCursor;
      reads++;
    } while (reads < 100);
    expect(combined).toBe(original);
    expect(reads).toBeGreaterThan(0);
    expect(store.details(saved.id, cursor)?.events.items).toEqual([]);
  });

  it('recovers interrupted runs without losing their existing logs', () => {
    const project = createProject();
    const previous = new BuildHistoryStore();
    previous.activate(project);
    previous.saveRun(run('interrupted'));
    previous.appendRaw('interrupted', 'stderr', 'work before crash', '2026-11-16T12:00:00.000Z');
    const next = new BuildHistoryStore();
    next.activate(project);
    expect(next.list().runs[0]).toMatchObject({
      state: 'failed',
      error: 'Interrupted: editor closed before build completion',
      stages: [{ state: 'failed' }],
    });
    expect(next.details('interrupted')?.logs.items[0]).toMatchObject({ text: 'work before crash' });
  });

  it('keeps project histories separate and rejects arbitrary filesystem reads', () => {
    const a = createProject('a');
    const b = createProject('b');
    const store = new BuildHistoryStore();
    store.activate(a);
    store.saveRun(run('first', 'a'));
    store.activate(b);
    expect(store.list()).toMatchObject({ projectGuid: 'b', total: 0 });
    expect(store.details('first')).toBeNull();
    expect(() => store.details('../index.json')).toThrow('Invalid build history run ID');
    store.saveRun(run('second', 'b'));
    store.activate(a);
    expect(store.list().runs.map((item) => item.id)).toEqual(['first']);
  });

  it('prunes the oldest completed runs and their files after the last 100', () => {
    const project = createProject();
    const store = new BuildHistoryStore();
    store.activate(project);
    for (let i = 0; i < 101; i++) {
      const entry = run(`run-${i}`);
      entry.state = 'succeeded';
      store.saveRun(entry);
      store.appendRaw(entry.id, 'stdout', `log ${i}`, entry.startedAt);
    }
    expect(store.list()).toMatchObject({ total: 100 });
    expect(store.details('run-0')).toBeNull();
    expect(store.list(0, 2).runs.map((item) => item.id)).toEqual(['run-100', 'run-99']);
    expect(fs.existsSync(path.join(project.projectRoot, 'Saved', 'BuildHistory', 'logs', 'run-0.jsonl'))).toBe(false);
  });

  it('refuses a saved-directory symlink that escapes the project', () => {
    if (process.platform === 'win32') return;
    const project = createProject();
    const outside = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-history-outside-'));
    folders.push(outside);
    fs.symlinkSync(outside, path.join(project.projectRoot, 'Saved'), 'dir');
    const store = new BuildHistoryStore();
    store.activate(project);
    expect(store.list().storageError).toContain('Unsafe build history directory');
    store.saveRun(run('unpersisted'));
    expect(fs.readdirSync(outside)).toEqual([]);
  });

  it('archives normalized event stream and raw chunks through BuildMonitorService', () => {
    const project = createProject();
    const monitor = new BuildMonitorService();
    monitor.attachProject(project);
    monitor.observeCommand({
      revision: 1,
      state: 'building',
      configuration: 'Debug',
      runId: 'run-live',
      startedAt: '2026-11-16T12:00:00.000Z',
      diagnostics: [{ sequence: 1, severity: 'error', message: 'compiler error' }],
      buildRequired: false,
      reloadRequired: false,
      restartRequired: false,
      history: [],
    }, 'game-a');
    monitor.appendRaw(project, 'run-live', 'stdout', 'complete stdout');
    const finished = monitor.snapshot().runs[0];
    expect(finished.state).toBe('running');

    const restored = new BuildMonitorService();
    restored.attachProject(project);
    expect(restored.historyPage().runs[0]).toMatchObject({ id: 'run-live', state: 'failed' });
    expect(restored.historyDetails('run-live')?.logs.items[0].text).toBe('complete stdout');
    const events = restored.historyDetails('run-live')?.events.items as BuildMonitorEvent[];
    expect(events.map((event) => event.kind)).toEqual(['run-started', 'diagnostic']);
  });
});
