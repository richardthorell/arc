export type RemoteAssetImportPhase = 'idle' | 'running' | 'canceling' | 'completed' | 'failed' | 'canceled';

export type RemoteAssetImportProgress = {
  bytesReceived: number;
  totalBytes?: number;
};

export type RemoteAssetImportLifecycle = {
  phase: RemoteAssetImportPhase;
  operationId: number;
  progress?: RemoteAssetImportProgress;
  error?: string;
};

export const initialRemoteAssetImportLifecycle = (): RemoteAssetImportLifecycle => ({
  phase: 'idle',
  operationId: 0,
});

export const beginRemoteAssetImport = (
  state: RemoteAssetImportLifecycle,
  operationId = state.operationId + 1,
): RemoteAssetImportLifecycle => ({
  phase: 'running',
  operationId,
  progress: { bytesReceived: 0 },
});

export const updateRemoteAssetImportProgress = (
  state: RemoteAssetImportLifecycle,
  operationId: number,
  progress: RemoteAssetImportProgress,
): RemoteAssetImportLifecycle => {
  if (operationId !== state.operationId || (state.phase !== 'running' && state.phase !== 'canceling')) {
    return state;
  }

  const bytesReceived = Math.max(state.progress?.bytesReceived ?? 0, Math.max(0, progress.bytesReceived));
  const totalBytes =
    progress.totalBytes !== undefined && progress.totalBytes > 0 ? progress.totalBytes : state.progress?.totalBytes;

  return {
    ...state,
    progress: {
      bytesReceived: totalBytes === undefined ? bytesReceived : Math.min(bytesReceived, totalBytes),
      ...(totalBytes === undefined ? {} : { totalBytes }),
    },
  };
};

export const requestRemoteAssetImportCancellation = (state: RemoteAssetImportLifecycle): RemoteAssetImportLifecycle =>
  state.phase === 'running' ? { ...state, phase: 'canceling' } : state;

export const completeRemoteAssetImport = (
  state: RemoteAssetImportLifecycle,
  operationId: number,
): RemoteAssetImportLifecycle =>
  operationId === state.operationId && state.phase === 'running'
    ? {
        phase: 'completed',
        operationId,
        ...(state.progress === undefined ? {} : { progress: state.progress }),
      }
    : state;

export const cancelRemoteAssetImport = (
  state: RemoteAssetImportLifecycle,
  operationId: number,
): RemoteAssetImportLifecycle =>
  operationId === state.operationId && (state.phase === 'running' || state.phase === 'canceling')
    ? {
        phase: 'canceled',
        operationId,
        ...(state.progress === undefined ? {} : { progress: state.progress }),
      }
    : state;

export const failRemoteAssetImport = (
  state: RemoteAssetImportLifecycle,
  operationId: number,
  error: string,
): RemoteAssetImportLifecycle =>
  operationId === state.operationId && (state.phase === 'running' || state.phase === 'canceling')
    ? {
        phase: 'failed',
        operationId,
        error,
        ...(state.progress === undefined ? {} : { progress: state.progress }),
      }
    : state;
