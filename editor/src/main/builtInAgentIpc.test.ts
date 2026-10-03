import { describe, expect, it } from 'vitest';

import type { BuiltInAgentInvokeRequest } from '../common/builtInAgentTypes';

const validRequest = (method: unknown, params?: unknown): BuiltInAgentInvokeRequest => {
  if (typeof method !== 'string' || method.trim() === '') throw new Error('Built-in agent method is required');
  return { method, ...(params !== undefined ? { params } : {}) };
};

describe('built-in agent IPC contract', () => {
  it('preserves opaque harness parameters without introducing transport aliases', () => {
    expect(validRequest('scene.getEntity', { guid: 'entity-guid' })).toEqual({
      method: 'scene.getEntity',
      params: { guid: 'entity-guid' },
    });
  });

  it('rejects empty operation identities', () => {
    expect(() => validRequest('')).toThrow('Built-in agent method is required');
  });
});
