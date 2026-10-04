import { describe, expect, it } from 'vitest';

import { agentHarnessMethods, gatewayHttpMethods } from './agentHarnessContract';

describe('agent selection transport contract', () => {
  it('uses the same operation names for harness and external gateway routes', () => {
    expect(agentHarnessMethods).toEqual(expect.arrayContaining(['selection.set', 'selection.clear']));
    expect(gatewayHttpMethods['/api/v1/selection/set']).toBe('selection.set');
    expect(gatewayHttpMethods['/api/v1/selection/clear']).toBe('selection.clear');
  });
});
