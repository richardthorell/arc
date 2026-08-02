import { describe, expect, it } from 'vitest';

import { SourceControlService } from './sourceControlService';

describe('SourceControlService', () => {
  it('honors the configured provider without invoking Git', async () => {
    const service = new SourceControlService(
      () => process.cwd(),
      () => false,
    );
    const snapshot = await service.snapshot();
    expect(snapshot.available).toBe(false);
    expect(snapshot.error).toContain('disabled');
  });

  it('rejects repository path traversal before invoking Git', () => {
    const service = new SourceControlService(() => 'D:/project');
    expect(() => service.diff('../outside.txt')).toThrow('inside the project');
    expect(() => service.stage(['C:/outside.txt'])).toThrow('inside the project');
  });

  it('rejects reference argument injection and reports missing projects', async () => {
    const service = new SourceControlService(() => null);
    expect((await service.checkout('--detach;danger')).succeeded).toBe(false);
    expect((await service.snapshot()).available).toBe(false);
  });

  it('permits inspection but rejects mutations for read-only projects', async () => {
    const service = new SourceControlService(
      () => process.cwd(),
      () => true,
      () => false,
    );
    expect((await service.commit('Should not run')).error).toContain('read-only');
  });
});
