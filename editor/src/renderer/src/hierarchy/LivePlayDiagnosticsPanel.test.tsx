// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { LivePlayDiagnosticsSnapshot } from '../../../common/livePlayDiagnosticsTypes';
import { LivePlayDiagnosticsPanel } from './LivePlayDiagnosticsPanel';

afterEach(cleanup);

const snapshot: LivePlayDiagnosticsSnapshot = {
  sessionId: 'session-a',
  instanceId: 'instance-a',
  connection: 'connected',
  negotiatedCapabilities: ['session.observe', 'entity.inspect'],
  lastReceivedSequence: 20,
  lastHeartbeatAtMs: 1700000000000,
  lastRuntimeMessageAtMs: 1700000001000,
  runtimeState: 'running',
  runtimeTick: 180,
  runtimeRevision: 12,
  contentGeneration: 8,
  contentStatus: 'active',
  lastAckAtMs: 1700000001000,
  lastAckLatencyMs: 33,
  lastInspectionLatencyMs: 7,
  appliedCommands: 3,
  rejectedCommands: 1,
  droppedLogLines: 0,
  events: [
    { id: 1, timestampMs: 1700000000000, kind: 'log', level: 'info', message: 'Game started' },
    {
      id: 2,
      timestampMs: 1700000001000,
      kind: 'command',
      level: 'warning',
      message: 'Command rejected: stale_revision',
    },
  ],
};

describe('Live Play session diagnostics UI', () => {
  it('shows native telemetry and bounded event history with a warning filter', () => {
    render(<LivePlayDiagnosticsPanel snapshot={snapshot} />);

    expect(screen.getByRole('region', { name: 'Live Play session diagnostics' })).toBeInTheDocument();
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.getByText('180')).toBeInTheDocument();
    expect(screen.getByText('33 ms')).toBeInTheDocument();
    expect(screen.getByText('session.observe · entity.inspect')).toBeInTheDocument();
    expect(screen.getByText('Game started')).toBeInTheDocument();
    expect(screen.getByText('Command rejected: stale_revision')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: 'Warnings only' }));
    expect(screen.queryByText('Game started')).not.toBeInTheDocument();
    expect(screen.getByText('Command rejected: stale_revision')).toBeInTheDocument();
  });

  it('does not invent measurements before the runtime reports them', () => {
    render(<LivePlayDiagnosticsPanel snapshot={null} />);
    expect(screen.getByText(/Waiting for a connected session diagnostics snapshot/u)).toBeInTheDocument();
  });
});
