// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { ProfilerPanel } from './ProfilerPanel';
import type { ProfilerSnapshot } from './ProfilerPanel';

const sample = (): ProfilerSnapshot => ({
  timestampNanoseconds: 100,
  memory: {
    bytes: 4 * 1024 * 1024,
    softLimit: 8 * 1024 * 1024,
    hardLimit: 16 * 1024 * 1024,
    pressureEvents: 1,
    domains: [
      { domain: 'components', bytes: 1024, peakBytes: 2048, softLimit: 4096, hardLimit: 8192, pressure: false },
      { domain: 'streaming', bytes: 2048, peakBytes: 4096, softLimit: 4096, hardLimit: 8192, pressure: false },
    ],
    groups: [
      { domain: 'components', tag: 'world.components', worldId: 7, threadId: 11, stackId: 99, allocationCount: 3, bytes: 1024 },
      { domain: 'streaming', tag: 'assets.streaming', worldId: 0, threadId: 12, stackId: 0, allocationCount: 1, bytes: 2048 },
    ],
  },
  scheduler: {
    submitted: 4,
    completed: 3,
    stolen: 2,
    cancelled: 0,
    failed: 1,
    queued: 1,
    droppedEvents: 0,
    jobs: [{
      sequence: 1,
      name: 'render.frame',
      priority: 'critical',
      affinity: 'render',
      status: 'succeeded',
      threadId: 11,
      queuedNanoseconds: 10,
      startedNanoseconds: 20,
      completedNanoseconds: 1020,
    }],
  },
});

describe('ProfilerPanel', () => {
  afterEach(cleanup);

  it('renders scheduler metrics, memory domains, and task scopes', () => {
    render(<ProfilerPanel samples={[sample()]} />);
    const groups = within(screen.getByTestId('profiler-allocation-groups'));
    expect(screen.getByText('4.0 MiB')).toBeInTheDocument();
    expect(screen.getByText('render.frame')).toBeInTheDocument();
    expect(groups.getByText('world.components')).toBeInTheDocument();
    expect(screen.getByText('Work Steals')).toBeInTheDocument();
  });

  it('filters allocation groups by domain and tag', () => {
    render(<ProfilerPanel samples={[sample()]} />);
    const selects = screen.getAllByRole('combobox');
    fireEvent.change(selects[0], { target: { value: 'streaming' } });
    const groups = within(screen.getByTestId('profiler-allocation-groups'));
    expect(groups.queryByText('world.components')).not.toBeInTheDocument();
    expect(groups.getByText('assets.streaming')).toBeInTheDocument();
  });

  it('shows a connected-host waiting state without samples', () => {
    render(<ProfilerPanel samples={[]} />);
    expect(screen.getByText('Waiting for engine telemetry')).toBeInTheDocument();
  });
});
