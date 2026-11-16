// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { BuildHistoryDetail, BuildHistoryPage } from '../../../common/buildHistoryTypes';
import type { BuildMonitorEvent, BuildMonitorRun, BuildMonitorSnapshot } from '../../../common/buildMonitorTypes';
import { BuildDashboard } from './BuildDashboard';

const current: BuildMonitorRun = {
  id: 'current-run',
  source: 'workflow',
  projectGuid: 'test-project',
  action: 'build',
  configuration: 'Shipping',
  platform: 'windows',
  profile: 'steam',
  startedAt: '2026-11-16T10:00:00.000Z',
  state: 'running',
  stages: [
    { id: 'current-run:configure', name: 'configure', state: 'succeeded', durationMs: 420 },
    { id: 'current-run:compile', name: 'compile', state: 'running' },
    { id: 'current-run:package', name: 'package', state: 'pending' },
  ],
  warningCount: 1,
  errorCount: 0,
};
const previous: BuildMonitorRun = {
  ...current,
  id: 'older-run',
  action: 'cook',
  configuration: 'Debug',
  state: 'failed',
  startedAt: '2026-11-15T10:00:00.000Z',
  completedAt: '2026-11-15T10:00:12.000Z',
  durationMs: 12000,
  stages: [{ id: 'older-run:cook', name: 'cook', state: 'failed', durationMs: 12000 }],
  errorCount: 1,
  warningCount: 0,
};
const monitor: BuildMonitorSnapshot = {
  revision: 1,
  projectGuid: 'test-project',
  activeRunId: current.id,
  runs: [previous, current],
  events: [],
};
const page: BuildHistoryPage = {
  projectGuid: 'test-project',
  total: 2,
  offset: 0,
  runs: [current, previous],
};

const makeDetails = (run: BuildMonitorRun): BuildHistoryDetail => ({
  run,
  logs: {
    items: [{ stream: 'stdout', text: `build log for ${run.id}\n`, timestamp: run.startedAt }],
    nextCursor: 48,
    hasMore: false,
  },
  events: {
    items:
      run.id === 'older-run'
        ? [
            {
              kind: 'diagnostic',
              sequence: 2,
              runId: run.id,
              timestamp: run.startedAt,
              stageId: 'older-run:cook',
              diagnostic: {
                sequence: 1,
                severity: 'error',
                message: 'Asset cooker failed',
                file: 'Source/Scene.cpp',
                line: 42,
              },
            },
          ]
        : [],
    nextCursor: 32,
    hasMore: false,
  },
});

function mockApi(overrides: Record<string, unknown> = {}) {
  const handlers: Array<(event: BuildMonitorEvent) => void> = [];
  const api = {
    monitorSnapshot: vi.fn().mockResolvedValue(monitor),
    history: vi.fn().mockResolvedValue(page),
    historyDetails: vi
      .fn()
      .mockImplementation(async (runId: string) => makeDetails(runId === current.id ? current : previous)),
    onMonitorEvent: vi.fn((listener: (event: BuildMonitorEvent) => void) => {
      handlers.push(listener);
      return () => {
        const index = handlers.indexOf(listener);
        if (index >= 0) handlers.splice(index, 1);
      };
    }),
    ...overrides,
  };
  vi.stubGlobal('arc', { build: api });
  return { api, handlers };
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Build Monitor BM3', () => {
  it('shows a current build, measured stages and a selectable archived failure with diagnostics', async () => {
    const { api } = mockApi();
    const openDiagnostic = vi.fn();
    const close = vi.fn();
    render(<BuildDashboard snapshot={null} onClose={close} onOpenDiagnostic={openDiagnostic} />);

    expect(screen.getByRole('dialog', { name: 'Build monitor' })).toBeInTheDocument();
    expect(await screen.findByText('RUNNING NOW')).toBeInTheDocument();
    expect(await screen.findByText('build log for current-run')).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Build stages' })).toBeInTheDocument();
    expect(screen.getByText('configure')).toBeInTheDocument();
    expect(screen.getByText('compile')).toBeInTheDocument();
    expect(screen.getByText('package')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cancel build' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /cook · Debug/ }));
    await waitFor(() => expect(api.historyDetails).toHaveBeenCalledWith('older-run'));
    expect(await screen.findByText('build log for older-run')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Diagnostics' }));
    const diagnostic = await screen.findByRole('button', { name: /Asset cooker failed/ });
    fireEvent.click(diagnostic);
    expect(openDiagnostic).toHaveBeenCalledWith(expect.objectContaining({ file: 'Source/Scene.cpp', line: 42 }));

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(close).toHaveBeenCalledOnce();
  });

  it('supports bounded output paging and searching the loaded archived chunks', async () => {
    const first = makeDetails(current);
    first.logs.hasMore = true;
    const later = makeDetails(current);
    later.logs.items = [{ stream: 'stderr', text: 'error: archive page two\n', timestamp: current.startedAt }];
    later.logs.hasMore = false;
    later.logs.nextCursor = 101;
    const details = vi.fn().mockImplementation(async (_id: string, cursor = 0) => (cursor ? later : first));
    mockApi({ historyDetails: details });
    render(<BuildDashboard snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);

    expect(await screen.findByText('build log for current-run')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Load more' }));
    await waitFor(() => expect(details).toHaveBeenCalledWith(current.id, 48, 32));
    expect(await screen.findByText('error: archive page two')).toBeInTheDocument();

    fireEvent.change(screen.getByRole('textbox', { name: 'Search build output' }), { target: { value: 'page two' } });
    expect(screen.getByText('error: archive page two')).toBeInTheDocument();
    expect(screen.queryByText('build log for current-run')).not.toBeInTheDocument();

    fireEvent.change(screen.getByRole('combobox', { name: 'Stream' }), { target: { value: 'stdout' } });
    expect(screen.queryByText('error: archive page two')).not.toBeInTheDocument();
  });

  it('retains a selected historical run when another live build sends events', async () => {
    const { handlers, api } = mockApi();
    render(<BuildDashboard snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);
    await screen.findByText('build log for current-run');
    fireEvent.click(screen.getByRole('button', { name: /cook · Debug/ }));
    await screen.findByText('build log for older-run');
    expect(handlers).toHaveLength(1);
    handlers[0]({
      kind: 'stage-changed',
      sequence: 10,
      runId: current.id,
      timestamp: current.startedAt,
      stage: { id: 'current-run:compile', name: 'compile', state: 'succeeded' },
    });
    await waitFor(() => expect(api.monitorSnapshot).toHaveBeenCalledTimes(2));
    expect(screen.getByText('build log for older-run')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /cook · Debug/ })).toHaveAttribute('aria-current', 'true');
  });

  it('reports a storage error without crashing and shows an empty state', async () => {
    mockApi({
      monitorSnapshot: vi.fn().mockResolvedValue({ ...monitor, runs: [], activeRunId: null }),
      history: vi
        .fn()
        .mockResolvedValue({
          projectGuid: 'test-project',
          total: 0,
          offset: 0,
          runs: [],
          storageError: 'History unavailable',
        }),
    });
    render(<BuildDashboard snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('History unavailable');
    expect(screen.getByText('No build selected')).toBeInTheDocument();
    expect(screen.getByText('No recorded builds yet.')).toBeInTheDocument();
  });
});
