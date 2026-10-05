export type PlayLifecycleState = 'stopped' | 'playing' | 'paused';

export type PlayLifecycleCommand =
  | Readonly<{ kind: 'play' }>
  | Readonly<{ kind: 'pause' }>
  | Readonly<{ kind: 'stop' }>
  | Readonly<{ kind: 'step'; ticks: number }>;

export type PlayLifecycleIntent = Readonly<{
  command: 'play' | 'pause' | 'stop' | 'step';
  from: PlayLifecycleState;
  expectedState: PlayLifecycleState;
  ticks?: number;
  scope: 'play-world';
}>;

/**
 * Validates an agent Play request before it reaches the editor command boundary.
 * Execution remains owned by the normal editor Play commands; this model only
 * describes a legal transition and the Play World scope the harness must use.
 */
export function planPlayLifecycleCommand(
  state: PlayLifecycleState,
  command: PlayLifecycleCommand,
): PlayLifecycleIntent {
  switch (command.kind) {
    case 'play':
      if (state === 'playing') throw new Error('Play is already running');
      return {
        command: 'play',
        from: state,
        expectedState: 'playing',
        scope: 'play-world',
      };
    case 'pause':
      if (state !== 'playing') throw new Error('Pause requires a running Play World');
      return {
        command: 'pause',
        from: state,
        expectedState: 'paused',
        scope: 'play-world',
      };
    case 'stop':
      if (state === 'stopped') throw new Error('Play is already stopped');
      return {
        command: 'stop',
        from: state,
        expectedState: 'stopped',
        scope: 'play-world',
      };
    case 'step': {
      if (state !== 'paused') throw new Error('Step requires a paused Play World');
      if (!Number.isSafeInteger(command.ticks) || command.ticks <= 0) {
        throw new Error('Step ticks must be a positive safe integer');
      }
      return {
        command: 'step',
        from: state,
        expectedState: 'paused',
        ticks: command.ticks,
        scope: 'play-world',
      };
    }
  }
}
