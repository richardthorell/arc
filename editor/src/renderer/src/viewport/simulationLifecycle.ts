export type RuntimeSessionMode = 'play' | 'simulate';

export type RuntimeSessionState =
  | { phase: 'authoring' }
  | { phase: 'running'; mode: RuntimeSessionMode }
  | { phase: 'paused'; mode: RuntimeSessionMode };

export type RuntimeSessionCommand =
  | { type: 'start'; mode: RuntimeSessionMode }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'switch-mode'; mode: RuntimeSessionMode }
  | { type: 'stop' };

/**
 * Pure lifecycle transition model shared by Play and Simulate controls.
 *
 * Simulate is a runtime-world mode rather than an editor-camera state. Switching
 * between Play and Simulate therefore keeps the same isolated runtime session;
 * stopping is the only transition back to the authoring world.
 */
export const transitionRuntimeSession = (
  state: RuntimeSessionState,
  command: RuntimeSessionCommand,
): RuntimeSessionState => {
  switch (command.type) {
    case 'start':
      if (state.phase !== 'authoring') throw new Error('A runtime session is already active');
      return { phase: 'running', mode: command.mode };

    case 'pause':
      if (state.phase !== 'running') throw new Error('Only a running runtime session can be paused');
      return { phase: 'paused', mode: state.mode };

    case 'resume':
      if (state.phase !== 'paused') throw new Error('Only a paused runtime session can be resumed');
      return { phase: 'running', mode: state.mode };

    case 'switch-mode':
      if (state.phase === 'authoring') throw new Error('Cannot switch mode without an active runtime session');
      return { phase: state.phase, mode: command.mode };

    case 'stop':
      if (state.phase === 'authoring') throw new Error('No runtime session is active');
      return { phase: 'authoring' };
  }
};
