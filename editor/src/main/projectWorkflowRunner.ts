import { randomUUID } from 'node:crypto';
import type { BuildTargetGraph } from '../common/buildTargetTypes';
import {
  workflowPlanSchema,
  type WorkflowPlan,
  type WorkflowAuthority,
  type WorkflowArtifact,
  type WorkflowCapability,
  type WorkflowRun,
  type WorkflowStage,
} from '../common/projectWorkflowTypes';

export type WorkflowStageContext = {
  readonly runId: string;
  readonly plan: WorkflowPlan;
  readonly authority: WorkflowAuthority;
  readonly signal: AbortSignal;
  assertCurrent(): void;
  /** Called only by trusted native build adapters, never renderer IPC. */
  publishTargetGraph(graph: BuildTargetGraph): void;
  reportTargetArtifact(targetId: string): void;
  reportRequestedTargetSatisfied(): void;
};

export type WorkflowStageAdapter = {
  capability(plan: WorkflowPlan, authority: WorkflowAuthority): WorkflowCapability;
  /** Work stays private until publication. Cancellation must terminate only owned processes. */
  execute(context: WorkflowStageContext): Promise<void>;
  /** Releases private staging and owned resources after failure/cancellation. Must be idempotent. */
  rollback(context: WorkflowStageContext): Promise<void>;
};

export type WorkflowPublication = {
  /** Synchronous final commit, after authority/cancellation checks. Throwing must retain the prior output. */
  commit(context: WorkflowStageContext): WorkflowArtifact[];
};

function freezeSnapshot<T>(value: T): T {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freezeSnapshot(child);
    Object.freeze(value);
  }
  return value;
}

/** Trusted main-process adapters only. No shell commands or executors arrive over renderer IPC. */
export class ProjectWorkflowRunner {
  private active: { controller: AbortController; run: WorkflowRun } | undefined;
  private last: WorkflowRun | undefined;

  constructor(
    private readonly adapters: Partial<Record<WorkflowStage, WorkflowStageAdapter>>,
    private readonly isCurrent: (authority: WorkflowAuthority) => boolean,
    private readonly publication: WorkflowPublication,
    private readonly notify: (run: WorkflowRun) => void,
  ) {}

  snapshot(): WorkflowRun | undefined {
    return this.last ? structuredClone(this.last) : undefined;
  }

  capabilities(
    plan: WorkflowPlan,
    authority: WorkflowAuthority,
  ): { stage: WorkflowStage; capability: WorkflowCapability }[] {
    return plan.stages.map((stage) => ({
      stage,
      capability: this.adapters[stage]?.capability(structuredClone(plan), structuredClone(authority)) ?? {
        available: false,
        reason: 'No executable adapter is registered',
      },
    }));
  }

  cancel(runId: string): void {
    if (this.active?.run.id !== runId || this.active.controller.signal.aborted || this.active.run.state !== 'running')
      return;
    this.active.controller.abort();
    this.active.run.state = 'cancelling';
    this.emit(this.active.run);
  }

  async execute(planInput: WorkflowPlan, authorityInput: WorkflowAuthority): Promise<WorkflowRun> {
    if (this.active) throw new Error('A workflow is already running or draining cancelled work');
    const plan = workflowPlanSchema.parse(planInput);
    const authority = structuredClone(authorityInput);
    if (plan.projectGuid !== authority.projectGuid || !this.isCurrent(authority))
      throw new Error('Stale project authority');
    if (plan.destination === 'device' && !authority.target)
      throw new Error('Device workflow requires an explicit target identity');
    // Preflight the entire plan before creating files, compiling, or starting any process.
    for (const { stage, capability } of this.capabilities(plan, authority)) {
      if (!capability.available) throw new Error(`${stage}: ${capability.reason ?? 'Stage unavailable'}`);
    }
    const controller = new AbortController();
    const runStart = performance.now();
    const run: WorkflowRun = {
      id: randomUUID(),
      startedAt: new Date().toISOString(),
      plan,
      authority,
      state: 'running',
      artifacts: [],
      stages: plan.stages.map((stage) => ({ stage, state: 'pending' })),
    };
    const context: WorkflowStageContext = {
      runId: run.id,
      plan: freezeSnapshot(structuredClone(plan)),
      authority: freezeSnapshot(structuredClone(authority)),
      signal: controller.signal,
      assertCurrent: () => {
        if (controller.signal.aborted) throw new Error('Workflow cancelled');
        if (!this.isCurrent(authority)) throw new Error('Workflow authority changed; output was not published');
      },
      publishTargetGraph: (graph) => {
        if (controller.signal.aborted || run.state !== 'running') return;
        run.targetGraph = structuredClone(graph);
        this.emit(run);
      },
      reportTargetArtifact: (targetId) => {
        if (controller.signal.aborted || run.state !== 'running') return;
        const graph = run.targetGraph;
        const target = graph?.nodes.find((node) => node.id === targetId);
        if (!graph || !target || target.artifactObservedAt) return;
        target.artifactObservedAt = new Date().toISOString();
        graph.revision++;
        this.emit(run);
      },
      reportRequestedTargetSatisfied: () => {
        if (controller.signal.aborted || run.state !== 'running') return;
        const graph = run.targetGraph;
        const target = graph?.nodes.find((node) => node.id === graph.rootTargetId);
        if (!graph || !target || target.satisfiedAt) return;
        target.satisfiedAt = new Date().toISOString();
        graph.revision++;
        this.emit(run);
      },
    };
    const entered: WorkflowStageAdapter[] = [];
    let activeStageClock: number | undefined;
    this.active = { controller, run };
    this.emit(run);
    try {
      for (const stage of run.stages) {
        context.assertCurrent();
        activeStageClock = performance.now();
        stage.state = 'running';
        stage.startedAt = new Date().toISOString();
        this.emit(run);
        context.assertCurrent();
        const adapter = this.adapters[stage.stage]!;
        entered.push(adapter);
        await adapter.execute(context);
        context.assertCurrent();
        stage.completedAt = new Date().toISOString();
        stage.durationMs = Math.max(0, performance.now() - activeStageClock!);
        activeStageClock = undefined;
        stage.state = 'succeeded';
        this.emit(run);
      }
      context.assertCurrent();
      run.artifacts = this.publication.commit(context);
      run.state = 'succeeded';
    } catch (error) {
      run.state = controller.signal.aborted ? 'cancelled' : 'failed';
      run.error = error instanceof Error ? error.message : String(error);
      const running = run.stages.find((stage) => stage.state === 'running');
      if (running) {
        running.completedAt = new Date().toISOString();
        if (activeStageClock !== undefined) running.durationMs = Math.max(0, performance.now() - activeStageClock);
        running.state = run.state;
      }
      for (const adapter of entered.reverse()) {
        try {
          await adapter.rollback(context);
        } catch (cleanupError) {
          run.error += `; cleanup: ${String(cleanupError)}`;
        }
      }
    } finally {
      run.completedAt = new Date().toISOString();
      run.durationMs = Math.max(0, performance.now() - runStart);
      this.active = undefined;
      this.emit(run);
    }
    return structuredClone(run);
  }

  private emit(run: WorkflowRun): void {
    this.last = structuredClone(run);
    // An observer cannot invalidate a completed commit or interrupt rollback.
    try {
      this.notify(structuredClone(run));
    } catch {
      /* UI delivery is best-effort. */
    }
  }
}
