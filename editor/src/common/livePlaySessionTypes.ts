/** Public read-only status of an editor-owned external Development game. */
export type WindowsLivePlayState = 'stopped' | 'starting' | 'connected' | 'disconnected' | 'failed' | 'stopping';

export type WindowsLivePlaySnapshot = {
  state: WindowsLivePlayState;
  sessionId: string | null;
  instanceId: string | null;
  pid: number | null;
  lastReceivedSequence: number;
  error: string | null;
};
