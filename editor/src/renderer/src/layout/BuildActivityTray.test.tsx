// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { BuildMonitorEvent, BuildMonitorRun, BuildMonitorSnapshot } from '../../../common/buildMonitorTypes';
import { BuildActivityTray } from './BuildActivityTray';

const running: BuildMonitorRun = {
  id: 'build-1',
  source: 'command',
  projectGuid: 'project-1',
  action: 'build',
  configuration: 'RelWithDebInfo',
  platform: 'windows',
  startedAt: '2026-11-18T20:00:00.000Z',
  state: 'running',
  stages: [{ id: 'build-1:compile', name: 'Compile', state: 'running' }],
  warningCount: 0,
  errorCount: 0,
};

const makeSnapshot = (run: BuildMonitorRun | null): BuildMonitorSnapshot => ({
  revision: 1,
  projectGuid: 'project-1',
  activeRunId: run?.state === 'running' ? run.id : null,
  runs: run ? [run] : [],
  events: [],
});

const mockBuildApi = (initial: BuildMonitorSnapshot) => {
  let snapshot = initial;
  const listeners: Array<(event: BuildMonitorEvent) => void> = [];
  const api = {
    monitorSnapshot: vi.fn().mockImplementation(async () => snapshot),
    onMonitorEvent: vi.fn((handler: (event: BuildMonitorEvent) => void) => {
      listeners.push(handler);
      return () => {
        const index = listeners.indexOf(handler);
        if (index >= 0) listeners.splice(index, 1);
      };
    }),
    onState: vi.fn(() => () => undefined),
    execute: vi.fn().mockResolvedValue({}),
    workflowSnapshot: vi.fn(),
    cancelWorkflow: vi.fn(),
  };
  vi.stubGlobal('arc', { build: api });
  return {
    api,
    update: (next: BuildMonitorSnapshot, event: BuildMonitorEvent) => {
      snapshot = next;
      listeners.forEach((listener) => listener(event));
    },
    listeners,
  };
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('BuildActivityTray', () => {
  it('remains accessible when Builds is closed and opens the full view on demand', async () => {
    const { api } = mockBuildApi(makeSnapshot(running));
    const onOpenBuilds = vi.fn();
    render(<BuildActivityTray projectKey="project-1" onOpenBuilds={onOpenBuilds} />);

    const indicator = await screen.findByRole('button', { name: 'Build activity: Building · Compile' });
    expect(indicator).toBeInTheDocument();
    fireEvent.click(indicator);
    expect(screen.getByRole('dialog', { name: 'Build activity' })).toHaveTextContent('Running in background');
    fireEvent.click(screen.getByRole('button', { name: 'Open Builds' }));
    expect(onOpenBuilds).toHaveBeenCalledOnce();
    expect(screen.queryByRole('dialog', { name: 'Build activity' })).not.toBeInTheDocument();
    expect(api.execute).not.toHaveBeenCalled();
    expect(api.onMonitorEvent).toHaveBeenCalledOnce();
  });

  it('updates on build completion and offers Cancel only during an active build', async () => {
    const { api, update } = mockBuildApi(makeSnapshot(running));
    render(<BuildActivityTray projectKey="project-1" onOpenBuilds={vi.fn()} />);
    fireEvent.click(await screen.findByRole('button', { name: 'Build activity: Building · Compile' }));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(api.execute).toHaveBeenCalledWith({ action: 'cancel' }));

    const completed: BuildMonitorRun = {
      ...running,
      state: 'succeeded',
      completedAt: '2026-11-18T20:00:14.000Z',
      durationMs: 14000,
      stages: [{ id: 'build-1:compile', name: 'Compile', state: 'succeeded' }],
    };
    update(makeSnapshot(completed), {
      kind: 'run-completed',
      runId: completed.id,
      run: completed,
      sequence: 2,
      timestamp: completed.completedAt!,
    });

    expect(await screen.findByRole('button', { name: 'Build activity: Build succeeded' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Cancel' })).not.toBeInTheDocument();
  });

  it('does not accidentally cancel a build when the tray closes and reopens', async () => {
    const { api } = mockBuildApi(makeSnapshot(running));
    render(<BuildActivityTray projectKey="project-1" onOpenBuilds={vi.fn()} />);
    const indicator = await screen.findByRole('button', { name: 'Build activity: Building · Compile' });
    fireEvent.click(indicator);
    fireEvent.click(indicator);
    expect(screen.queryByRole('dialog', { name: 'Build activity' })).not.toBeInTheDocument();
    fireEvent.click(indicator);
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    expect(api.execute).not.toHaveBeenCalled();
  });
});
