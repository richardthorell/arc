import { AlertTriangle, Pause, Play, Square, StepForward } from 'lucide-react';

import type { CommandId, EditorRuntimeState } from '../app/workbenchTypes';
import { UiIconButton } from '../ui';

import './playSessionBar.css';

export type PlaySessionBarProps = {
  state: Exclude<EditorRuntimeState, 'stopped'>;
  error?: string;
  tickId: number;
  onCommand: (command: CommandId) => void;
};

export function PlaySessionBar({ state, error, tickId, onCommand }: PlaySessionBarProps) {
  const stateLabel = `${state[0].toUpperCase()}${state.slice(1)}`;
  return (
    <section className={`play-session-bar is-${state}`} aria-label="Play session">
      <span className="play-session-world">Play World</span>
      <strong>{stateLabel}</strong>
      <span className="play-session-tick">Tick {tickId.toLocaleString()}</span>
      {error && (
        <span className="play-session-error" role="alert" title={error}>
          <AlertTriangle aria-hidden="true" size={13} />
          {error}
        </span>
      )}
      <span className="play-session-spacer" />
      <UiIconButton
        disabled={state === 'running' || state === 'faulted'}
        label={state === 'paused' ? 'Resume Play World' : 'Play'}
        onClick={() => onCommand('scene.play')}
        variant="toolbar"
      >
        <Play aria-hidden="true" fill="currentColor" size={12} strokeWidth={0} />
      </UiIconButton>
      <UiIconButton
        active={state === 'paused'}
        disabled={state !== 'running'}
        label="Pause Play World"
        onClick={() => onCommand('scene.pause')}
        variant="toolbar"
      >
        <Pause aria-hidden="true" size={12} />
      </UiIconButton>
      <UiIconButton label="Stop Play World" onClick={() => onCommand('scene.stop')} variant="toolbar">
        <Square aria-hidden="true" size={11} />
      </UiIconButton>
      <UiIconButton
        disabled={state !== 'paused'}
        label="Step Play World"
        onClick={() => onCommand('scene.step')}
        variant="toolbar"
      >
        <StepForward aria-hidden="true" size={12} />
      </UiIconButton>
    </section>
  );
}
