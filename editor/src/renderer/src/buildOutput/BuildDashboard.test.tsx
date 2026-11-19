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
    execute: vi.fn().mockResolvedValue({}),
    executeWorkflow: vi.fn().mockResolvedValue({}),
    revealArtifact: vi.fn().mockResolvedValue(''),
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

describe('Build Monitor workspace presentation', () => {
  it('renders modelessly inside the document workspace without trapping Escape or cancelling the build', async () => {
    const { api } = mockApi();
    const onClose = vi.fn();
    const onCancel = vi.fn();
    render(
      <BuildDashboard
        presentation="workspace"
        snapshot={null}
        onClose={onClose}
        onCancel={onCancel}
        onOpenDiagnostic={vi.fn()}
      />,
    );

    expect(screen.getByRole('region', { name: 'Build monitor' })).toBeInTheDocument();
    expect(screen.queryByRole('dialog', { name: 'Build monitor' })).not.toBeInTheDocument();
    expect(await screen.findByText('build log for current-run')).toBeInTheDocument();
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Close build monitor' })).not.toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
    expect(onCancel).not.toHaveBeenCalled();
    expect(api.monitorSnapshot).toHaveBeenCalled();
  });
});

describe('Build Monitor command controls', () => {
  it('allows starting a code build with the project-selected configuration when idle', async () => {
    const { api } = mockApi({
      monitorSnapshot: vi.fn().mockResolvedValue({ ...monitor, activeRunId: null, runs: [previous] }),
      history: vi.fn().mockResolvedValue({ ...page, runs: [previous] }),
    });
    render(
      <BuildDashboard
        presentation="workspace"
        snapshot={{
          revision: 1,
          state: 'idle',
          configuration: 'Debug',
          buildRequired: true,
          reloadRequired: false,
          restartRequired: false,
          diagnostics: [],
        }}
        onClose={vi.fn()}
        onOpenDiagnostic={vi.fn()}
      />,
    );
    await screen.findByText('build log for older-run');
    fireEvent.click(screen.getByRole('button', { name: 'Start code build' }));
    await waitFor(() => expect(api.execute).toHaveBeenCalledWith({ action: 'build', configuration: 'Debug' }));
  });

  it('disables new code builds while a workflow is already running', async () => {
    mockApi();
    render(<BuildDashboard snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);
    await screen.findByText('build log for current-run');
    expect(screen.getByRole('button', { name: 'Start code build' })).toBeDisabled();
  });
});

describe('Build Monitor actionable workflow', () => {
  it('shows a failed command reason with zero compiler diagnostics and retries the recorded configuration', async () => {
    const failed: BuildMonitorRun = {
      ...previous,
      id: 'failed-command',
      source: 'command',
      action: 'build',
      configuration: 'RelWithDebInfo',
      state: 'failed',
      errorCount: 0,
      error: 'CMake configuration failed: no matching ARC SDK',
      stages: [{ id: 'failed-command:compile', name: 'Compile', state: 'failed' }],
    };
    const { api } = mockApi({
      monitorSnapshot: vi.fn().mockResolvedValue({ ...monitor, activeRunId: null, runs: [failed] }),
      history: vi.fn().mockResolvedValue({ ...page, total: 1, runs: [failed] }),
      historyDetails: vi.fn().mockResolvedValue(makeDetails(failed)),
    });
    render(<BuildDashboard presentation="workspace" snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);
    expect(await screen.findByRole('region', { name: 'Build result details' })).toHaveTextContent(
      'CMake configuration failed: no matching ARC SDK',
    );
    expect(screen.getByText('No compiler diagnostics recorded')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Retry selected build' }));
    await waitFor(() => expect(api.execute).toHaveBeenCalledWith({ action: 'build', configuration: 'RelWithDebInfo' }));
    expect(screen.getByRole('button', { name: 'View error output' })).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Target dependency graph' })).not.toBeInTheDocument();
  });

  it('filters build history and safely requests reveal for published workflow artifacts', async () => {
    const built: BuildMonitorRun = {
      ...current,
      platform: 'android',
      targetId: 'android-arm64-vulkan',
      profile: 'android-debug',
      configuration: 'Debug',
      state: 'succeeded',
      completedAt: '2026-11-16T10:02:00.000Z',
      durationMs: 120000,
      artifacts: [
        {
          kind: 'package',
          path: 'Build/Packages/Android/game.apk',
          manifestPath: 'Build/Packages/Android/distribution.json',
          buildId: 'current-run',
          targetId: 'android-arm64-vulkan',
          profile: 'android-debug',
        },
      ],
      request: {
        action: 'build',
        profile: 'android-debug',
        targetId: 'android-arm64-vulkan',
        destination: 'none',
        sessionId: '',
      },
    };
    const { api } = mockApi({
      monitorSnapshot: vi.fn().mockResolvedValue({ ...monitor, activeRunId: null, runs: [built] }),
      history: vi.fn().mockResolvedValue({ ...page, runs: [built, previous] }),
      historyDetails: vi.fn().mockImplementation(async (id: string) => makeDetails(id === built.id ? built : previous)),
    });
    render(<BuildDashboard snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);

    expect(await screen.findByRole('region', { name: 'Build artifacts' })).toHaveTextContent(
      'Build/Packages/Android/game.apk',
    );
    fireEvent.click(screen.getByRole('button', { name: 'Reveal package artifact' }));
    await waitFor(() =>
      expect(api.revealArtifact).toHaveBeenCalledWith('current-run', 'Build/Packages/Android/game.apk'),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Retry selected build' }));
    await waitFor(() => expect(api.executeWorkflow).toHaveBeenCalledWith(built.request));
    fireEvent.change(screen.getByRole('combobox', { name: 'Filter build status' }), { target: { value: 'failed' } });
    expect(screen.getByRole('button', { name: /cook · Debug/ })).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Filter build platform' }), { target: { value: 'android' } });
    expect(screen.getByText('No builds match these filters.')).toBeInTheDocument();
  });
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

  it('drills into a measured source and filters only text matched in the selected build log', async () => {
    const traced: BuildMonitorRun = {
      ...current,
      source: 'command',
      compileTrace: {
        source: 'ninja-log',
        clock: 'ninja-invocation',
        truncated: false,
        tasks: [
          {
            output: 'CMakeFiles/MyEditor.dir/main.cpp.o',
            sourceFile: 'Source/main.cpp',
            targetName: 'MyEditor',
            startMs: 0,
            endMs: 60,
            durationMs: 60,
          },
        ],
      },
    };
    const details = makeDetails(traced);
    details.logs.items = [
      { stream: 'stdout', text: 'CMakeFiles/MyEditor.dir/main.cpp.o built\n', timestamp: traced.startedAt },
      { stream: 'stdout', text: 'unrelated compiler output\n', timestamp: traced.startedAt },
    ];
    mockApi({
      monitorSnapshot: vi.fn().mockResolvedValue({ ...monitor, runs: [previous, traced] }),
      history: vi.fn().mockResolvedValue({ ...page, runs: [traced, previous] }),
      historyDetails: vi.fn().mockResolvedValue(details),
    });
    render(<BuildDashboard snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);
    expect(await screen.findByText('unrelated compiler output')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Target dependencies and compile performance'));
    fireEvent.click(screen.getByRole('tab', { name: 'Diagnostics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Inspect compile task: Source/main.cpp' }));
    expect(screen.getByText('MyEditor', { selector: 'dd' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Find output text in build logs' }));
    expect(screen.getByRole('tab', { name: 'Output' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('textbox', { name: 'Search build output' })).toHaveValue(
      'CMakeFiles/MyEditor.dir/main.cpp.o',
    );
    expect(screen.getByText('CMakeFiles/MyEditor.dir/main.cpp.o built')).toBeInTheDocument();
    expect(screen.queryByText('unrelated compiler output')).not.toBeInTheDocument();
  });

  it('refreshes measured live tasks when the compile-trace event is published', async () => {
    let currentMonitor = monitor;
    const monitorSnapshot = vi.fn().mockImplementation(async () => currentMonitor);
    const { handlers } = mockApi({ monitorSnapshot });
    render(<BuildDashboard snapshot={null} onClose={vi.fn()} onOpenDiagnostic={vi.fn()} />);
    await screen.findByText('build log for current-run');
    expect(screen.queryByRole('region', { name: 'Compilation task timeline' })).not.toBeInTheDocument();

    const traced: BuildMonitorRun = {
      ...current,
      source: 'command',
      compileTrace: {
        source: 'ninja-log',
        clock: 'ninja-invocation',
        truncated: false,
        tasks: [{ output: 'CMakeFiles/game.dir/live.cpp.o', startMs: 0, endMs: 80, durationMs: 80 }],
      },
    };
    currentMonitor = { ...monitor, revision: monitor.revision + 1, runs: [previous, traced] };
    handlers[0]({
      kind: 'compile-trace-changed',
      runId: current.id,
      timestamp: current.startedAt,
      sequence: 12,
      source: 'ninja-log',
      taskCount: 1,
    });
    await waitFor(() => expect(monitorSnapshot).toHaveBeenCalledTimes(2));
    fireEvent.click(await screen.findByText('Target dependencies and compile performance'));
    expect(await screen.findByText('1 measured · Ninja · live')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Inspect compile task: CMakeFiles/game.dir/live.cpp.o' }),
    ).toBeInTheDocument();
  });

  it('reports a storage error without crashing and shows an empty state', async () => {
    mockApi({
      monitorSnapshot: vi.fn().mockResolvedValue({ ...monitor, runs: [], activeRunId: null }),
      history: vi.fn().mockResolvedValue({
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
