import { execFile } from 'node:child_process';
import {
  workflowPlanSchema,
  workflowRequestSchema,
  type WorkflowRequest,
  type WorkflowPlan,
} from '../common/projectWorkflowTypes';

/** Read-only CLI planning. Native policy is shared with non-editor tooling. */
export async function queryProjectWorkflowPlan(
  tool: string,
  descriptorPath: string,
  request: WorkflowRequest,
): Promise<WorkflowPlan> {
  const input = workflowRequestSchema.parse(request);
  const args = [
    'workflow',
    'plan',
    '--json',
    '--project',
    descriptorPath,
    '--profile',
    input.profile,
    '--target',
    input.targetId,
    '--action',
    input.action,
    '--destination',
    input.destination,
  ];
  if (input.sessionId) args.push('--session-id', input.sessionId);
  const stdout = await new Promise<string>((resolve, reject) => {
    execFile(
      tool,
      args,
      { shell: false, windowsHide: true, timeout: 30_000, maxBuffer: 1024 * 1024, encoding: 'utf8' },
      (error, output) => {
        if (error) {
          try {
            const failure = JSON.parse(output) as { error?: unknown };
            if (typeof failure.error === 'string') return reject(new Error(failure.error));
          } catch {
            /* Preserve the process error when the CLI did not return structured diagnostics. */
          }
          reject(error);
        } else resolve(output);
      },
    );
  });
  const plan = workflowPlanSchema.parse(JSON.parse(stdout));
  if (
    plan.action !== input.action ||
    plan.profile !== input.profile ||
    plan.targetId !== input.targetId ||
    plan.destination !== input.destination ||
    plan.sessionId !== input.sessionId
  )
    throw new Error('Native workflow plan does not match the captured request');
  return plan;
}
