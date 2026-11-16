import type { ArcBuildDiagnostic } from './buildTypes';
import type { WorkflowArtifact } from './projectWorkflowTypes';

/** Unified, project-scoped build observation contract. BM1 is in-memory; BM2 adds disk persistence. */
export type BuildMonitorSource = 'command' | 'workflow';
export type BuildMonitorResult = 'running' | 'succeeded' | 'failed' | 'cancelled';
export type BuildMonitorStageState = 'pending' | BuildMonitorResult;

export type BuildMonitorStage = {
  id: string;
  name: string;
  state: BuildMonitorStageState;
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
};

export type BuildMonitorRun = {
  id: string;
  source: BuildMonitorSource;
  projectGuid: string;
  action: string;
  configuration: string;
  profile?: string;
  targetId?: string;
  platform?: string;
  startedAt: string;
  completedAt?: string;
  durationMs?: number;
  state: BuildMonitorResult;
  stages: BuildMonitorStage[];
  warningCount: number;
  errorCount: number;
  error?: string;
  artifacts?: WorkflowArtifact[];
};

type EventBase = { sequence: number; runId: string; timestamp: string };
export type BuildMonitorEvent =
  | (EventBase & { kind: 'run-started'; run: BuildMonitorRun })
  | (EventBase & { kind: 'stage-changed'; stage: BuildMonitorStage })
  | (EventBase & { kind: 'output'; text: string })
  | (EventBase & { kind: 'diagnostic'; diagnostic: ArcBuildDiagnostic })
  | (EventBase & { kind: 'run-completed'; run: BuildMonitorRun });

export type BuildMonitorSnapshot = {
  revision: number;
  projectGuid: string | null;
  activeRunId: string | null;
  runs: BuildMonitorRun[];
  /** Bounded replay buffer for renderer reconnect; not durable build history. */
  events: BuildMonitorEvent[];
};
