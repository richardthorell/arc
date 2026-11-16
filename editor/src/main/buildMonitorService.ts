import type { ArcBuildSnapshot } from '../common/buildTypes';
import type { BuildMonitorEvent, BuildMonitorRun, BuildMonitorStage, BuildMonitorStageState, BuildMonitorSnapshot } from '../common/buildMonitorTypes';
import type { WorkflowExecutionSnapshot } from '../common/projectWorkflowTypes';

const MAX_RUNS = 24;
const MAX_EVENTS = 512;
const commandStages: Record<string, string> = {
  configuring: 'Configure',
  building: 'Compile',
  cleaning: 'Clean',
};
const terminal = (state: string): state is 'succeeded' | 'failed' | 'cancelled' =>
  state === 'succeeded' || state === 'failed' || state === 'cancelled';
const elapsed = (start: string, end: string): number =>
  Math.max(0, Date.parse(end) - Date.parse(start));

/**
 * Normalizes existing command and project-workflow snapshots into one event stream.
 * Does not infer CMake target dependencies or completion percentages. Those need
 * tool-level instrumentation (BM4/BM5). Logs and runs are bounded until BM2.
 */
export class BuildMonitorService {
  private value: BuildMonitorSnapshot = {
    revision: 1, projectGuid: null, activeRunId: null, runs: [], events: [],
  };
  private sequence = 0;
  private diagnosticCursor = new Map<string, number>();
  private outputCursor = new Map<string, number>();

  constructor(
    private readonly publish: (event: BuildMonitorEvent) => void = () => {},
    private readonly now: () => string = () => new Date().toISOString(),
  ) {}

  snapshot(): BuildMonitorSnapshot {
    return structuredClone(this.value);
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

  private event(event: Omit<BuildMonitorEvent, 'sequence' | 'timestamp'>): void {
    const next = { ...event, sequence: ++this.sequence, timestamp: this.now() } as BuildMonitorEvent;
    this.value.events = [...this.value.events.slice(-(MAX_EVENTS - 1)), structuredClone(next)];
    this.value.revision++;
    try {
      this.publish(structuredClone(next));
    } catch {
      // A renderer disappearing must never interrupt native compilation.
    }
  }

  private register(run: BuildMonitorRun): BuildMonitorRun {
    this.value.runs = [...this.value.runs.slice(-(MAX_RUNS - 1)), run];
    this.value.activeRunId = run.id;
    this.event({ kind: 'run-started', runId: run.id, run: structuredClone(run) });
    return run;
  }

  private find(id: string): BuildMonitorRun | undefined {
    return this.value.runs.find((run) => run.id === id);
  }

  private updateStage(run: BuildMonitorRun, stage: BuildMonitorStage, state: BuildMonitorStageState): void {
    if (stage.state === state) return;
    const timestamp = this.now();
    stage.state = state;
    if (state === 'running') stage.startedAt = timestamp;
    if (terminal(state)) {
      stage.completedAt = timestamp;
      if (stage.startedAt) stage.durationMs = elapsed(stage.startedAt, timestamp);
    }
    this.event({ kind: 'stage-changed', runId: run.id, stage: structuredClone(stage) });
  }

  private complete(run: BuildMonitorRun, result: 'succeeded' | 'failed' | 'cancelled', endedAt?: string): void {
    if (terminal(run.state)) return;
    run.state = result;
    run.completedAt = endedAt ?? this.now();
    run.durationMs = elapsed(run.startedAt, run.completedAt);
    if (this.value.activeRunId === run.id) this.value.activeRunId = null;
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
        action: stageName.toLowerCase(),
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
          ? { kind: 'output', runId: run.id, text: diagnostic.message }
          : { kind: 'diagnostic', runId: run.id, diagnostic: structuredClone(diagnostic) },
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
      this.complete(run, completed.result, snapshot.completedAt);
      run.durationMs = completed.durationMs;
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
        startedAt: this.now(),
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
      if (stage) this.updateStage(run, stage, source.state);
    }

    const output = snapshot.latestOutput;
    if (output && output.sequence > (this.outputCursor.get(run.id) ?? 0)) {
      this.outputCursor.set(run.id, output.sequence);
      this.event({ kind: 'output', runId: run.id, text: output.text });
    }
    if (terminal(workflow.state)) {
      run.error = workflow.error;
      run.artifacts = structuredClone(workflow.artifacts);
      this.complete(run, workflow.state);
    }
  }
}
