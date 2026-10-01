import { describe, expect, it } from 'vitest';

import {
  FLOW_INSTANCE_OVERRIDE_SCHEMA_VERSION,
  parseFlowInstanceOverrides,
  serializeFlowInstanceOverrides,
} from './flowInstanceOverridePersistence';

describe('flow instance override persistence', () => {
  it('round-trips per-instance overrides by stable variable id', () => {
    const persisted = serializeFlowInstanceOverrides([
      { variableId: 'speed', value: 4.5 },
      { variableId: 'enabled', value: true },
    ]);

    expect(persisted).toEqual({
      version: FLOW_INSTANCE_OVERRIDE_SCHEMA_VERSION,
      overrides: [
        { variableId: 'speed', value: 4.5 },
        { variableId: 'enabled', value: true },
      ],
    });
    expect(parseFlowInstanceOverrides(persisted)).toEqual({ ok: true, overrides: persisted.overrides });
  });

  it('keeps independent entity payloads independent', () => {
    const first = serializeFlowInstanceOverrides([{ variableId: 'speed', value: 2 }]);
    const second = serializeFlowInstanceOverrides([{ variableId: 'speed', value: 8 }]);

    expect(first.overrides[0]?.value).toBe(2);
    expect(second.overrides[0]?.value).toBe(8);
  });

  it('rejects unsupported versions and malformed payloads atomically', () => {
    expect(parseFlowInstanceOverrides({ version: 2, overrides: [] }).ok).toBe(false);
    expect(parseFlowInstanceOverrides({ version: 1, overrides: [{ variableId: '', value: 1 }] }).ok).toBe(false);
    expect(parseFlowInstanceOverrides({ version: 1, overrides: 'invalid' }).ok).toBe(false);
  });

  it('rejects duplicate stable variable ids', () => {
    const result = parseFlowInstanceOverrides({
      version: 1,
      overrides: [
        { variableId: 'speed', value: 1 },
        { variableId: 'speed', value: 2 },
      ],
    });

    expect(result).toEqual({ ok: false, error: 'Duplicate Flow instance override: speed' });
  });
});
