import type { BuiltInAgentToolExecutionResult } from '../../../common/builtInAgentTypes';
import type { AiToolCall } from '../../../common/aiRuntimeTypes';
import type { AiAgentToolInvoker } from './aiAgentToolLoop';

export type AiAgentApprovalMode = 'ask' | 'auto';
export type AiAgentApprovalDecision = 'approved' | 'denied';

export type AiAgentApprovalRequest = Readonly<{
  id: string;
  clientId: string;
  clientName: string;
  label: string;
  requestedAt: string;
  state: 'pending' | 'approved' | 'denied' | 'expired';
  expiresAt?: string;
}>;

type PendingApproval = {
  request: AiAgentApprovalRequest;
  result: BuiltInAgentToolExecutionResult;
  resolve: (result: BuiltInAgentToolExecutionResult) => void;
  reject: (error: Error) => void;
  detachAbort: () => void;
};

type AiAgentApprovalCoordinatorOptions = Readonly<{
  invokeTool: AiAgentToolInvoker;
  approve: (requestId: string) => Promise<boolean>;
  deny: (requestId: string) => Promise<boolean>;
}>;

const parseApprovalRequest = (result: BuiltInAgentToolExecutionResult): AiAgentApprovalRequest => {
  let value: unknown;
  try {
    value = JSON.parse(result.content) as unknown;
  } catch {
    throw new Error('ARC edit approval request returned invalid JSON');
  }
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('ARC edit approval request returned an invalid payload');

  const request = value as Partial<AiAgentApprovalRequest>;
  if (
    typeof request.id !== 'string' ||
    typeof request.clientId !== 'string' ||
    typeof request.clientName !== 'string' ||
    typeof request.label !== 'string' ||
    typeof request.requestedAt !== 'string' ||
    (request.state !== 'pending' &&
      request.state !== 'approved' &&
      request.state !== 'denied' &&
      request.state !== 'expired')
  ) {
    throw new Error('ARC edit approval request is missing required fields');
  }
  return request as AiAgentApprovalRequest;
};

const resultWithDecision = (
  result: BuiltInAgentToolExecutionResult,
  request: AiAgentApprovalRequest,
  state: AiAgentApprovalDecision,
): BuiltInAgentToolExecutionResult => {
  const content = JSON.stringify({ ...request, state });
  return {
    ...result,
    content,
    truncated: false,
    originalBytes: new TextEncoder().encode(content).byteLength,
  };
};

const abortedError = () => new Error('AI edit approval was cancelled');

export class AiAgentApprovalCoordinator {
  private mode: AiAgentApprovalMode = 'ask';
  private readonly pending = new Map<string, PendingApproval>();

  constructor(private readonly options: AiAgentApprovalCoordinatorOptions) {}

  setMode(mode: AiAgentApprovalMode): void {
    this.mode = mode;
    if (mode === 'auto') {
      for (const requestId of [...this.pending.keys()]) void this.approve(requestId);
    }
  }

  async invokeTool(call: AiToolCall, signal?: AbortSignal): Promise<BuiltInAgentToolExecutionResult> {
    const result = await this.options.invokeTool(call, signal);
    if (call.name !== 'edit.request') return result;

    const request = parseApprovalRequest(result);
    if (request.state !== 'pending') return result;
    if (signal?.aborted) throw abortedError();

    if (this.mode === 'auto') {
      if (!(await this.options.approve(request.id))) throw new Error('ARC edit approval could not be granted');
      return resultWithDecision(result, request, 'approved');
    }

    return new Promise<BuiltInAgentToolExecutionResult>((resolve, reject) => {
      const onAbort = () => {
        const pending = this.pending.get(request.id);
        if (!pending) return;
        this.pending.delete(request.id);
        pending.detachAbort();
        reject(abortedError());
      };
      if (signal) signal.addEventListener('abort', onAbort, { once: true });
      this.pending.set(request.id, {
        request,
        result,
        resolve,
        reject,
        detachAbort: () => signal?.removeEventListener('abort', onAbort),
      });
    });
  }

  async approve(requestId: string): Promise<boolean> {
    const approved = await this.options.approve(requestId);
    if (approved) this.settle(requestId, 'approved');
    return approved;
  }

  async deny(requestId: string): Promise<boolean> {
    const denied = await this.options.deny(requestId);
    if (denied) this.settle(requestId, 'denied');
    return denied;
  }

  dispose(): void {
    for (const [requestId, pending] of this.pending) {
      this.pending.delete(requestId);
      pending.detachAbort();
      pending.reject(abortedError());
    }
  }

  private settle(requestId: string, decision: AiAgentApprovalDecision): void {
    const pending = this.pending.get(requestId);
    if (!pending) return;
    this.pending.delete(requestId);
    pending.detachAbort();
    pending.resolve(resultWithDecision(pending.result, pending.request, decision));
  }
}
