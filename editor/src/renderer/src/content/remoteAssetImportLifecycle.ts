export type RemoteAssetImportPhase = 'idle' | 'running' | 'canceling' | 'completed' | 'failed' | 'canceled';

export type RemoteAssetImportLifecycle = {
  phase: RemoteAssetImportPhase;
  operationId: number;
  error?: string;
};

export const initialRemoteAssetImportLifecycle = (): RemoteAssetImportLifecycle => ({
  phase: 'idle',
  operationId: 0,
});

export const beginRemoteAssetImport = (state: RemoteAssetImportLifecycle): RemoteAssetImportLifecycle => ({
  phase: 'running',
  operationId: state.operationId + 1,
});

export const requestRemoteAssetImportCancellation = (state: RemoteAssetImportLifecycle): RemoteAssetImportLifecycle =>
  state.phase === 'running' ? { ...state, phase: 'canceling' } : state;

export const completeRemoteAssetImport = (
  state: RemoteAssetImportLifecycle,
  operationId: number,
): RemoteAssetImportLifecycle =>
  operationId === state.operationId && state.phase === 'running' ? { phase: 'completed', operationId } : state;

export const cancelRemoteAssetImport = (
  state: RemoteAssetImportLifecycle,
  operationId: number,
): RemoteAssetImportLifecycle =>
  operationId === state.operationId && (state.phase === 'running' || state.phase === 'canceling')
    ? { phase: 'canceled', operationId }
    : state;

export const failRemoteAssetImport = (
  state: RemoteAssetImportLifecycle,
  operationId: number,
  error: string,
): RemoteAssetImportLifecycle =>
  operationId === state.operationId && (state.phase === 'running' || state.phase === 'canceling')
    ? { phase: 'failed', operationId, error }
    : state;
