import type { LivePlayCapability } from './livePlayProtocol';
import type { WindowsLivePlayState } from './livePlaySessionTypes';

/** Bounded, sanitized, read-only diagnostics for one external Development instance. */
export type LivePlayDiagnosticEntry = {
  id: number;
  timestampMs: number;
  kind: 'connection' | 'runtime' | 'log' | 'command' | 'content' | 'warning';
  level: 'debug' | 'info' | 'warning' | 'error';
  message: string;
};

export type LivePlayDiagnosticsSnapshot = {
  sessionId: string | null;
  instanceId: string | null;
  connection: WindowsLivePlayState;
  negotiatedCapabilities: LivePlayCapability[];
  lastReceivedSequence: number;
  lastHeartbeatAtMs: number | null;
  lastRuntimeMessageAtMs: number | null;
  runtimeState: 'starting' | 'running' | 'paused' | 'stopped' | 'faulted' | null;
  runtimeTick: number | null;
  runtimeRevision: number | null;
  contentGeneration: number | null;
  contentStatus: 'available' | 'active' | 'rejected' | null;
  lastAckAtMs: number | null;
  lastAckLatencyMs: number | null;
  lastInspectionLatencyMs: number | null;
  appliedCommands: number;
  rejectedCommands: number;
  droppedLogLines: number;
  events: LivePlayDiagnosticEntry[];
};
