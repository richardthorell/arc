import { describe, expect, it } from 'vitest';

import { agentEditActions } from './agentHarnessContract';

describe('agent harness contract', () => {
  it('advertises prefab operations through the authoritative edit action catalog', () => {
    expect(agentEditActions).toContain('createPrefab');
    expect(agentEditActions).toContain('instantiatePrefab');
  });
});
