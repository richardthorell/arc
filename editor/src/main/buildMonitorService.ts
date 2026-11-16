import type { ArcBuildSnapshot } from '../common/buildTypes';
import type { ArcProjectCandidate } from '../common/projectTypes';
import type { BuildHistoryPage, BuildHistoryDetail } from '../common/buildHistoryTypes';
import { BuildHistoryStore } from './buildHistoryStore';
import type {
  BuildMonitorEvent,
  BuildMonitorRun,
  BuildMonitorStage,
  BuildMonitorStageState,
  BuildMonitorSnapshot,
} from '../common/buildMonitorTypes';
import type { WorkflowExecutionSnapshot, WorkflowStageProgress } from '../common/projectWorkflowTypes';

const MAX_RUNS = 24;
const MAX_EVENTS = 512;
const commandStages: Record<string, string> = {
  configuring: 'Configure',
  building: 'Compile',
  cleaning: 'Clean',
};
const terminal = (state: string): state is 'succeeded' | 'failed' | 'cancelled' =>
  state === 'succeeded' || state === 'failed' || state === 'cancelled';
const elapsed = (start: string, end: string): number => Math.max(0, Date.parse(end) - Date.parse(start));

/**
 * Normalizes existing command and project-workflow snapshots into one event stream.
 * Does not infer CMake target dependencies or completion percentages. Those need
 * tool-level instrumentation (BM4/BM5). The live buffer remains bounded;
 * BM2 persists completed runs, structured events and raw process output.
 */
type UnstampedEvent<T> = T extends BuildMonitorEvent ? Omit<T, 'sequence' | 'timestamp'> : never;
type BuildMonitorEventInput = UnstampedEvent<BuildMonitorEvent>;

export class BuildMonitorService {
  private value: BuildMonitorSnapshot = {
    revision: 1,
    projectGuid: null,
    activeRunId: null,
    runs: [],
    events: [],
  };
  private sequence = 0;
  private diagnosticCursor = new Map<string, number>();
  private outputCursor = new Map<string, number>();
  private readonly history = new BuildHistoryStore();

  constructor(
    private readonly publish: (event: BuildMonitorEvent) => void = () => {},
    private readonly now: () => string = () => new Date().toISOString(),
  ) {}

  snapshot(): BuildMonitorSnapshot {
    return structuredClone(this.value);
  }

  attachProject(project: ArcProjectCandidate | null): void {
    if (this.history.activate(project)) {
      // A different project root may share the same GUID; never reuse its live events.
      this.resetProject(null);
      this.resetProject(project?.descriptor.guid ?? null);
    }
  }

  historyPage(offset = 0, limit = 50): BuildHistoryPage {
    return this.history.list(offset, limit);
  }

  historyDetails(runId: string, logCursor = 0, eventCursor = 0): BuildHistoryDetail | null {
    return this.history.details(runId, logCursor, eventCursor);
  }

  appendRaw(project: ArcProjectCandidate, runId: string, stream: 'stdout' | 'stderr', text: string): void {
    if (project.descriptor.guid !== this.value.projectGuid) return;
    this.history.appendRaw(runId, stream, text, this.now());
  }

  resetProject(projectGuid: string | null): void {
    if (projectGuid === this.value.projectGuid) return;
    this.value = {
      revision: this.value.revision + 1,
      projectGuid,
      activeRunId: null,
      runs: [],
      events: [],
    };
    this.sequence = 0;
    this.diagnosticCursor.clear();
    this.outputCursor.clear();
  }

  private event(event: BuildMonitorEventInput): void {
    const next = { ...event, sequence: ++this.sequence, timestamp: this.now() } as BuildMonitorEvent;
    this.value.events = [...this.value.events.slice(-(MAX_EVENTS - 1)), structuredClone(next)];
    this.value.revision++;
    this.history.saveEvent(next);
    try {
      this.publish(structuredClone(next));
    } catch {
      // A renderer disappearing must never interrupt native compilation.
    }
  }

  private register(run: BuildMonitorRun): BuildMonitorRun {
    this.value.runs = [...this.value.runs.slice(-(MAX_RUNS - 1)), run];
    this.value.activeRunId = run.id;
    this.history.saveRun(run);
    this.event({ kind: 'run-started', runId: run.id, run: structuredClone(run) });
    return run;
  }

  private find(id: string): BuildMonitorRun | undefined {
    return this.value.runs.find((run) => run.id === id);
  }

  private updateStage(
    run: BuildMonitorRun,
    stage: BuildMonitorStage,
    state: BuildMonitorStageState,
    measured?: WorkflowStageProgress,
  ): void {
    // Recover the runner's timestamps even if a transient event was missed.
    // The legacy command path still observes the main-process clock.
    const timestamp = this.now();
    const startedAt = measured?.startedAt ?? (state === 'running' ? (stage.startedAt ?? timestamp) : stage.startedAt);
    const completedAt = measured?.completedAt ?? (terminal(state) ? (stage.completedAt ?? timestamp) : stage.completedAt);
    const durationMs = measured?.durationMs ??
      (completedAt && startedAt ? elapsed(startedAt, completedAt) : stage.durationMs);
    if (
      stage.state === state &&
      stage.startedAt === startedAt &&
      stage.completedAt === completedAt &&
      stage.durationMs === durationMs
    ) return;
    stage.state = state;
    stage.startedAt = startedAt;
    stage.completedAt = completedAt;
    stage.durationMs = durationMs;
    this.history.saveRun(run);
    this.event({ kind: 'stage-changed', runId: run.id, stageId: stage.id, stage: structuredClone(stage) });
  }

  private complete(
    run: BuildMonitorRun,
    result: 'succeeded' | 'failed' | 'cancelled',
    endedAt?: string,
    durationMs?: number,
  ): void {
    if (terminal(run.state)) return;
    run.state = result;
    run.completedAt = endedAt ?? this.now();
    run.durationMs = durationMs ?? elapsed(run.startedAt, run.completedAt);
    if (this.value.activeRunId === run.id) this.value.activeRunId = null;
    this.history.saveRun(run);
    this.event({ kind: 'run-completed', runId: run.id, run: structuredClone(run) });
  }

  observeCommand(snapshot: ArcBuildSnapshot, projectGuid: string | null): void {
    this.resetProject(projectGuid);
    if (!projectGuid || !snapshot.runId) return;
    const active = snapshot.state in commandStages;
    const finished = terminal(snapshot.state);
    if (!active && !finished) return;
    let run = this.find(snapshot.runId);
    if (!run) {
      const startedAt = snapshot.startedAt ?? this.now();
      const stageName = commandStages[snapshot.state] ?? 'Command';
      run = this.register({
        id: snapshot.runId,
        source: 'command',
        projectGuid,
        action: snapshot.state === 'building' ? 'build' : snapshot.state === 'cleaning' ? 'clean' : 'configure',
        configuration: snapshot.configuration,
        state: 'running',
        startedAt,
        stages: [{ id: `${snapshot.runId}:command`, name: stageName, state: 'running', startedAt }],
        warningCount: 0,
        errorCount: 0,
      });
    }

    const cursor = this.diagnosticCursor.get(run.id) ?? 0;
    for (const diagnostic of snapshot.diagnostics) {
      if (diagnostic.sequence <= cursor) continue;
      this.event(
        diagnostic.severity === 'info'
          ? { kind: 'output', runId: run.id, stageId: run.stages[0].id, text: diagnostic.message }
          : { kind: 'diagnostic', runId: run.id, stageId: run.stages[0].id, diagnostic: structuredClone(diagnostic) },
      );
    }
    this.diagnosticCursor.set(run.id, Math.max(cursor, ...snapshot.diagnostics.map((item) => item.sequence), 0));
    run.warningCount = snapshot.diagnostics.filter((item) => item.severity === 'warning').length;
    run.errorCount = snapshot.diagnostics.filter((item) => item.severity === 'error').length;

    if (finished) {
      // The legacy service sends a cancel state before its child exits. The
      // history record is the authoritative terminal event and exit duration.
      const completed = snapshot.history?.find((item) => item.runId === run!.id);
      if (!completed) return;
      this.updateStage(run, run.stages[0], completed.result);
      this.complete(run, completed.result, snapshot.completedAt, completed.durationMs);
    }
  }

  observeWorkflow(snapshot: WorkflowExecutionSnapshot, projectGuid: string | null): void {
    this.resetProject(projectGuid);
    const workflow = snapshot.run;
    if (!projectGuid || !workflow || workflow.plan.projectGuid !== projectGuid) return;

    let run = this.find(workflow.id);
    if (!run) {
      run = this.register({
        id: workflow.id,
        source: 'workflow',
        projectGuid,
        action: workflow.plan.action,
        configuration: workflow.plan.configuration,
        profile: workflow.plan.profile,
        targetId: workflow.plan.targetId,
        platform: workflow.plan.platform,
        state: 'running',
        startedAt: workflow.startedAt ?? this.now(),
        stages: workflow.stages.map((stage) => ({ 
          id: `${workflow.id}:${stage.stage}`,
          name: stage.stage,
          state: 'pending',
        })),
        warningCount: 0,
        errorCount: 0,
      });
    }

    // Workflow stages come from the native planner and runner, never a UI guess.
    for (const source of workflow.stages) {
      const stage = run.stages.find((entry) => entry.name === source.stage);
      if (stage) this.updateStage(run, stage, source.state, source);
    }

    const output = snapshot.latestOutput;
    if (output && output.sequence > (this.outputCursor.get(run.id) ?? 0)) {
      this.outputCursor.set(run.id, output.sequence);
      this.event({
        kind: 'output',
        runId: run.id,
        stageId: run.stages.find((stage) => stage.state === 'running')?.id,
        text: output.text,
      });
    }
    if (terminal(workflow.state)) {
      run.error = workflow.error;
      run.artifacts = structuredClone(workflow.artifacts);
      this.complete(run, workflow.state, workflow.completedAt, workflow.durationMs);
    }
  }
}
