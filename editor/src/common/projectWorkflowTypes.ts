import { z } from 'zod';
import type { BuildTargetGraph } from './buildTargetTypes';

export const workflowStageSchema = z.enum([
  'validate',
  'snapshot',
  'configure',
  'compile',
  'cook',
  'package',
  'deploy',
  'launch',
  'connect',
]);
export type WorkflowStage = z.infer<typeof workflowStageSchema>;

export const workflowRequestSchema = z
  .object({
    action: z.enum(['configure', 'buildCode', 'cook', 'build', 'play', 'runBuilt']),
    profile: z.string().min(1).max(256),
    targetId: z.string().min(1).max(256),
    destination: z.enum(['none', 'local', 'device']),
    sessionId: z
      .string()
      .max(64)
      .regex(/^[A-Za-z0-9-]*$/u),
  })
  .strict();
export type WorkflowRequest = z.infer<typeof workflowRequestSchema>;

/** Native workflow plan is the authority for stage selection; never infer stages in the UI. */
export const workflowPlanSchema = workflowRequestSchema
  .extend({
    succeeded: z.literal(true),
    version: z.literal(1),
    projectGuid: z.string().min(1),
    engineVersion: z.string().min(1),
    configuration: z.string().min(1),
    platform: z.string().min(1),
    architecture: z.string().min(1),
    renderer: z.string().min(1),
    finalizer: z.string().min(1),
    cookOutput: z.string().min(1),
    packageOutput: z.string().min(1),
    stages: z.array(workflowStageSchema).min(1).max(9),
  })
  .strict()
  .superRefine((plan, context) => {
    if (plan.stages[0] !== 'validate' || new Set(plan.stages).size !== plan.stages.length)
      context.addIssue({ code: 'custom', message: 'Invalid or duplicate workflow stages' });
  });
export type WorkflowPlan = z.infer<typeof workflowPlanSchema>;

export type WorkflowAuthority = {
  projectGuid: string;
  projectRoot: string;
  descriptorPath: string;
  projectRevision: number;
  sdkIdentity: string;
  target?: { sdkRoot: string; id: string };
};
export type WorkflowArtifact = {
  kind: 'code' | 'cooked' | 'package';
  path: string;
  manifestPath: string;
  buildId: string;
  targetId: string;
  profile: string;
};
export type WorkflowCapability = { available: boolean; reason?: string };
export type WorkflowStageProgress = {
  stage: WorkflowStage;
  state: 'pending' | 'running' | 'succeeded' | 'failed' | 'cancelled';
  /** Times are produced by the runner at stage boundaries, never inferred by the renderer. */
  startedAt?: string;
  completedAt?: string;
  /** Monotonic duration measured at execution time. */
  durationMs?: number;
};

export type WorkflowRun = {
  id: string;
  /** The runner owns the wall-clock boundaries and monotonic duration. */
  startedAt?: string;
  completedAt?: string;
  durationMs?: number;
  plan: WorkflowPlan;
  authority: WorkflowAuthority;
  state: 'running' | 'cancelling' | 'succeeded' | 'failed' | 'cancelled';
  stages: WorkflowStageProgress[];
  artifacts: WorkflowArtifact[];
  /** CMake File API graph for the requested target, when supported. */
  targetGraph?: BuildTargetGraph;
  error?: string;
};

export type WorkflowExecutionSnapshot = {
  requestId: string | null;
  planning: boolean;
  run: WorkflowRun | null;
  output: string[];
  /** Monotonic cursor for exact output delivery even after the bounded buffer rolls over. */
  latestOutput?: { sequence: number; text: string };
  error: string | null;
};

export type CookWorkflowOptions = {
  authority: string;
  profiles: Array<{
    id: string;
    targetId: string;
    platform: string;
    label: string;
    /** Capability of the current Editor executor, not of the standalone CLI. */
    editorBuildAvailable?: boolean;
    editorBuildReason?: string;
  }>;
  unavailableReason?: string;
};
