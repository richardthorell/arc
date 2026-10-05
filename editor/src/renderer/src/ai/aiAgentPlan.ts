import type {
  AiJsonObject,
  AiTaskProgress,
  AiTaskProgressState,
  AiToolCall,
  AiToolDefinition,
  AiToolResult,
} from '../../../common/aiRuntimeTypes';

export const AI_AGENT_PLAN_TOOL_NAME = 'agent.updatePlan';

const planStates: readonly AiTaskProgressState[] = [
  'planned',
  'in_progress',
  'completed',
  'failed',
  'cancelled',
] as const;

const planStepSchema: AiJsonObject = {
  type: 'object',
  additionalProperties: false,
  required: ['id', 'title', 'state'],
  properties: {
    id: { type: 'string', minLength: 1 },
    title: { type: 'string', minLength: 1 },
    state: { type: 'string', enum: [...planStates] },
    detail: { type: 'string' },
    children: {
      type: 'array',
      maxItems: 8,
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'title', 'state'],
        properties: {
          id: { type: 'string', minLength: 1 },
          title: { type: 'string', minLength: 1 },
          state: { type: 'string', enum: [...planStates] },
          detail: { type: 'string' },
        },
      },
    },
  },
};

export const aiAgentPlanToolDefinition: AiToolDefinition = {
  name: AI_AGENT_PLAN_TOOL_NAME,
  description:
    'Publish or update a semantic execution plan for multi-step ARC work. Reuse the same planId and stable step ids across updates. Use concise user-facing titles, mark exactly the active step in_progress, and retain completed/failed/cancelled steps instead of deleting history.',
  inputSchema: {
    type: 'object',
    additionalProperties: false,
    required: ['planId', 'title', 'steps'],
    properties: {
      planId: { type: 'string', minLength: 1 },
      title: { type: 'string', minLength: 1 },
      steps: {
        type: 'array',
        minItems: 1,
        maxItems: 12,
        items: planStepSchema,
      },
    },
  },
};

type ParsedStep = Readonly<{
  id: string;
  title: string;
  state: AiTaskProgressState;
  detail?: string;
  children?: readonly ParsedStep[];
}>;

type ParsedPlan = Readonly<{
  planId: string;
  title: string;
  steps: readonly ParsedStep[];
}>;

const isObject = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const stringValue = (value: unknown, label: string): string => {
  if (typeof value !== 'string' || !value.trim()) throw new Error(`${label} must be a non-empty string`);
  return value.trim();
};

const stateValue = (value: unknown, label: string): AiTaskProgressState => {
  if (typeof value !== 'string' || !planStates.includes(value as AiTaskProgressState))
    throw new Error(`${label} must be one of ${planStates.join(', ')}`);
  return value as AiTaskProgressState;
};

const parseStep = (value: unknown, label: string, allowChildren: boolean): ParsedStep => {
  if (!isObject(value)) throw new Error(`${label} must be an object`);
  const childrenValue = value.children;
  if (!allowChildren && childrenValue !== undefined)
    throw new Error(`${label}.children cannot be nested more than one level`);
  if (childrenValue !== undefined && !Array.isArray(childrenValue)) throw new Error(`${label}.children must be an array`);
  if (Array.isArray(childrenValue) && childrenValue.length > 8)
    throw new Error(`${label}.children may contain at most 8 steps`);
  const detail = value.detail === undefined ? undefined : stringValue(value.detail, `${label}.detail`);
  return {
    id: stringValue(value.id, `${label}.id`),
    title: stringValue(value.title, `${label}.title`),
    state: stateValue(value.state, `${label}.state`),
    ...(detail ? { detail } : {}),
    ...(Array.isArray(childrenValue)
      ? {
          children: childrenValue.map((child, index) => parseStep(child, `${label}.children[${index}]`, false)),
        }
      : {}),
  };
};

export const parseAgentPlan = (argumentsValue: AiJsonObject): ParsedPlan => {
  const raw = argumentsValue as Record<string, unknown>;
  if (!Array.isArray(raw.steps) || !raw.steps.length) throw new Error('steps must contain at least one plan step');
  if (raw.steps.length > 12) throw new Error('steps may contain at most 12 plan steps');
  const plan: ParsedPlan = {
    planId: stringValue(raw.planId, 'planId'),
    title: stringValue(raw.title, 'title'),
    steps: raw.steps.map((step, index) => parseStep(step, `steps[${index}]`, true)),
  };
  const ids = new Set<string>([plan.planId]);
  const visit = (steps: readonly ParsedStep[]) => {
    for (const step of steps) {
      if (ids.has(step.id)) throw new Error(`Plan task id '${step.id}' is duplicated`);
      ids.add(step.id);
      if (step.children) visit(step.children);
    }
  };
  visit(plan.steps);
  return plan;
};

const flattenParsedSteps = (steps: readonly ParsedStep[]): ParsedStep[] => {
  const result: ParsedStep[] = [];
  const collect = (entries: readonly ParsedStep[]) => {
    for (const entry of entries) {
      result.push(entry);
      if (entry.children) collect(entry.children);
    }
  };
  collect(steps);
  return result;
};

const planState = (steps: readonly ParsedStep[]): AiTaskProgressState => {
  const flat = flattenParsedSteps(steps);
  if (flat.some((step) => step.state === 'failed')) return 'failed';
  if (flat.some((step) => step.state === 'in_progress')) return 'in_progress';
  if (flat.length && flat.every((step) => step.state === 'completed')) return 'completed';
  if (flat.length && flat.every((step) => step.state === 'cancelled')) return 'cancelled';
  return 'planned';
};

const taskForStep = (step: ParsedStep, planId: string, agentStep: number): AiTaskProgress => ({
  id: step.id,
  title: step.title,
  state: step.state,
  agentStep,
  planId,
  ...(step.detail ? { detail: step.detail } : {}),
  ...(step.children?.length
    ? { children: step.children.map((child) => taskForStep(child, planId, agentStep)) }
    : {}),
});

export const taskTreeForAgentPlan = (plan: ParsedPlan, agentStep: number): AiTaskProgress => ({
  id: plan.planId,
  title: plan.title,
  state: planState(plan.steps),
  agentStep,
  planId: plan.planId,
  children: plan.steps.map((step) => taskForStep(step, plan.planId, agentStep)),
});

export const flattenTaskTree = (root: AiTaskProgress): AiTaskProgress[] => {
  const result: AiTaskProgress[] = [root];
  const collect = (children: readonly AiTaskProgress[] | undefined) => {
    for (const child of children ?? []) {
      result.push(child);
      collect(child.children);
    }
  };
  collect(root.children);
  return result;
};

export const mapTaskTree = (
  root: AiTaskProgress,
  taskId: string,
  update: (task: AiTaskProgress) => AiTaskProgress,
): AiTaskProgress => {
  if (root.id === taskId) return update(root);
  if (!root.children?.length) return root;
  return {
    ...root,
    children: root.children.map((child) => mapTaskTree(child, taskId, update)),
  };
};

export const agentPlanToolResult = (call: AiToolCall, taskCount: number): AiToolResult => ({
  toolCallId: call.id,
  name: call.name,
  operation: call.name,
  content: JSON.stringify({ accepted: true, taskCount }),
});
