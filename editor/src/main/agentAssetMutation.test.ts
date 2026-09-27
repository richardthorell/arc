import { describe, expect, it } from 'vitest';

import { assetRevision, parseEditableAgentAsset, prepareAgentAssetMutation } from './agentAssetMutation';

const material = {
  version: 4,
  name: 'Test',
  graph: { version: 1, nodes: [], connections: [] },
};

const flow = {
  version: 1,
  assetType: 'flow',
  name: 'Test Flow',
  graph: { version: 1, variables: [], nodes: [], connections: [] },
};

const json = (value: unknown): string => `${JSON.stringify(value, null, 2)}\n`;

describe('agent asset mutation contracts', () => {
  it('returns a stable content revision with an editable snapshot', () => {
    const contents = json(material);
    const snapshot = parseEditableAgentAsset('material', 'Materials/Test.arcmat', contents);
    expect(snapshot.revision).toBe(assetRevision(contents));
    expect(snapshot.definition).toEqual(material);
  });

  it('rejects a stale revision instead of silently overwriting', () => {
    const contents = json(material);
    expect(() =>
      prepareAgentAssetMutation(contents, {
        kind: 'material',
        path: 'Materials/Test.arcmat',
        expectedRevision: 'sha256:stale',
        definition: { ...material, name: 'Changed' },
      }),
    ).toThrow(/revision conflict/);
  });

  it('validates the replacement through the typed material contract', () => {
    const contents = json(material);
    expect(() =>
      prepareAgentAssetMutation(contents, {
        kind: 'material',
        path: 'Materials/Test.arcmat',
        expectedRevision: assetRevision(contents),
        definition: { version: 4, graph: { version: 1, nodes: [] } },
      }),
    ).toThrow(/version-1 graph/);
  });

  it('prepares a deterministic Flow replacement and new revision', () => {
    const contents = json(flow);
    const changed = { ...flow, name: 'Updated Flow' };
    const result = prepareAgentAssetMutation(contents, {
      kind: 'flow',
      path: 'Flow/Test.arcflow',
      expectedRevision: assetRevision(contents),
      definition: changed,
    });
    expect(result.changed).toBe(true);
    expect(result.contents).toBe(json(changed));
    expect(result.revision).toBe(assetRevision(result.contents));
  });

  it('rejects mismatched asset extensions', () => {
    expect(() => parseEditableAgentAsset('flow', 'Flow/Test.arcmat', json(flow))).toThrow(/\.arcflow/);
  });
});
