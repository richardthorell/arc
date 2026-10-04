import { describe, expect, it } from 'vitest';

import { agentEditActions } from './agentHarnessContract';

describe('editor selection authority boundary', () => {
  it('keeps selection out of persistent edit actions', () => {
    expect(agentEditActions).not.toContain('selection.set');
    expect(agentEditActions).not.toContain('selection.clear');
  });
});
