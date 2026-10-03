import { describe, expect, it } from 'vitest';

import {
  beginRemoteAssetImport,
  cancelRemoteAssetImport,
  completeRemoteAssetImport,
  failRemoteAssetImport,
  initialRemoteAssetImportLifecycle,
  requestRemoteAssetImportCancellation,
  updateRemoteAssetImportProgress,
} from './remoteAssetImportLifecycle';

describe('remote asset import lifecycle', () => {
  it('tracks cancellation explicitly until the worker acknowledges it', () => {
    const running = beginRemoteAssetImport(initialRemoteAssetImportLifecycle());
    const canceling = requestRemoteAssetImportCancellation(running);
    expect(canceling.phase).toBe('canceling');
    expect(cancelRemoteAssetImport(canceling, running.operationId).phase).toBe('canceled');
  });

  it('ignores stale completion from an earlier import', () => {
    const first = beginRemoteAssetImport(initialRemoteAssetImportLifecycle());
    const second = beginRemoteAssetImport(first);
    expect(completeRemoteAssetImport(second, first.operationId)).toEqual(second);
    expect(completeRemoteAssetImport(second, second.operationId).phase).toBe('completed');
  });

  it('does not report success after cancellation has been requested', () => {
    const running = beginRemoteAssetImport(initialRemoteAssetImportLifecycle());
    const canceling = requestRemoteAssetImportCancellation(running);
    expect(completeRemoteAssetImport(canceling, running.operationId)).toEqual(canceling);
  });

  it('keeps actionable failure text for the active operation only', () => {
    const running = beginRemoteAssetImport(initialRemoteAssetImportLifecycle());
    const failed = failRemoteAssetImport(running, running.operationId, 'checksum mismatch');
    expect(failed).toMatchObject({ phase: 'failed', error: 'checksum mismatch' });
    expect(failRemoteAssetImport(running, running.operationId - 1, 'stale')).toEqual(running);
  });

  it('tracks monotonic byte progress for the active operation', () => {
    const running = beginRemoteAssetImport(initialRemoteAssetImportLifecycle());
    const progressed = updateRemoteAssetImportProgress(running, running.operationId, {
      bytesReceived: 64,
      totalBytes: 128,
    });
    expect(progressed.progress).toEqual({ bytesReceived: 64, totalBytes: 128 });
    expect(updateRemoteAssetImportProgress(progressed, running.operationId, { bytesReceived: 32 }).progress).toEqual({
      bytesReceived: 64,
      totalBytes: 128,
    });
  });

  it('ignores stale progress and clamps progress to the known total', () => {
    const running = beginRemoteAssetImport(initialRemoteAssetImportLifecycle());
    expect(updateRemoteAssetImportProgress(running, running.operationId - 1, { bytesReceived: 50 })).toEqual(running);
    expect(
      updateRemoteAssetImportProgress(running, running.operationId, { bytesReceived: 256, totalBytes: 128 }).progress,
    ).toEqual({ bytesReceived: 128, totalBytes: 128 });
  });

  it('preserves the last progress snapshot while cancellation is pending and after termination', () => {
    const running = beginRemoteAssetImport(initialRemoteAssetImportLifecycle());
    const progressed = updateRemoteAssetImportProgress(running, running.operationId, { bytesReceived: 48 });
    const canceling = requestRemoteAssetImportCancellation(progressed);
    const later = updateRemoteAssetImportProgress(canceling, running.operationId, { bytesReceived: 64 });
    expect(later.progress).toEqual({ bytesReceived: 64 });
    expect(cancelRemoteAssetImport(later, running.operationId).progress).toEqual({ bytesReceived: 64 });
  });
});
