import { describe, expect, it } from 'vitest';

import {
  beginRemoteAssetImport,
  cancelRemoteAssetImport,
  completeRemoteAssetImport,
  failRemoteAssetImport,
  initialRemoteAssetImportLifecycle,
  requestRemoteAssetImportCancellation,
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
});
