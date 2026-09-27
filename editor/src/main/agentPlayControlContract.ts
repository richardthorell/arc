export const agentPlayActions = ['play', 'resume', 'pause', 'stop', 'step'] as const;

export type AgentPlayAction = (typeof agentPlayActions)[number];

export type AgentPlayState = 'stopped' | 'playing' | 'paused';

export type AgentPlayStatus = {
  state: AgentPlayState;
  fixedTick: number;
  worldEpoch: number;
  lastError: string | null;
};

export type AgentPlayControlRequest = {
  action: AgentPlayAction;
  ticks?: number;
};

const maximumStepTicks = 1024;

export const validateAgentPlayControlRequest = (
  request: AgentPlayControlRequest,
  state: AgentPlayState,
): AgentPlayControlRequest => {
  if (!agentPlayActions.includes(request.action)) {
    throw new Error(`Unsupported Play action: ${String(request.action)}`);
  }

  if (request.action === 'step') {
    if (state !== 'paused') {
      throw new Error('Play step requires a paused Play World');
    }
    const ticks = request.ticks ?? 1;
    if (!Number.isSafeInteger(ticks) || ticks < 1 || ticks > maximumStepTicks) {
      throw new Error(`Play step ticks must be an integer from 1 to ${maximumStepTicks}`);
    }
    return { action: 'step', ticks };
  }

  if (request.ticks !== undefined) {
    throw new Error('ticks is only valid for the step action');
  }

  if (request.action === 'play' && state !== 'stopped') {
    throw new Error('Play can only start from a stopped state');
  }
  if (request.action === 'resume' && state !== 'paused') {
    throw new Error('Resume requires a paused Play World');
  }
  if (request.action === 'pause' && state !== 'playing') {
    throw new Error('Pause requires a running Play World');
  }
  if (request.action === 'stop' && state === 'stopped') {
    throw new Error('Stop requires an active Play World');
  }

  return { action: request.action };
};
