import { describe, expect, it } from 'vitest';

import type { FlowVariableDefinition } from './flowGraphTypes';
import {
  reconcileFlowVariableOverrides,
  resetFlowVariableOverride,
  resolveFlowVariableOverrides,
  setFlowVariableOverride,
} from './flowVariableOverrides';

const variables: FlowVariableDefinition[] = [
  { id: 'speed', name: 'Speed', type: 'float', defaultValue: 3, exposed: true },
  { id: 'title', name: 'Title', type: 'string', defaultValue: 'Default', exposed: true },
  { id: 'internal', name: 'Internal', type: 'bool', defaultValue: false, exposed: false },
];

describe('flow variable overrides', () => {
  it('resolves exposed variables against defaults without exposing graph-local variables', () => {
    expect(resolveFlowVariableOverrides(variables, [{ variableId: 'speed', value: 8 }])).toEqual([
      {
        id: 'speed',
        name: 'Speed',
        type: 'float',
        defaultValue: 3,
        value: 8,
        overridden: true,
      },
      {
        id: 'title',
        name: 'Title',
        type: 'string',
        defaultValue: 'Default',
        value: 'Default',
        overridden: false,
      },
    ]);
  });

  it('updates existing overrides in place and appends newly authored overrides deterministically', () => {
    const existing = [
      { variableId: 'speed', value: 5 },
      { variableId: 'title', value: 'One' },
    ];

    expect(setFlowVariableOverride(existing, 'speed', 9)).toEqual([
      { variableId: 'speed', value: 9 },
      { variableId: 'title', value: 'One' },
    ]);
    expect(setFlowVariableOverride(existing, 'new', true)).toEqual([
      ...existing,
      { variableId: 'new', value: true },
    ]);
  });

  it('resets one override back to the graph default', () => {
    expect(
      resetFlowVariableOverride(
        [
          { variableId: 'speed', value: 5 },
          { variableId: 'title', value: 'One' },
        ],
        'speed',
      ),
    ).toEqual([{ variableId: 'title', value: 'One' }]);
  });

  it('reconciles overrides by stable variable id after graph edits', () => {
    expect(
      reconcileFlowVariableOverrides(variables, [
        { variableId: 'speed', value: 5 },
        { variableId: 'removed', value: 10 },
        { variableId: 'internal', value: true },
        { variableId: 'speed', value: 7 },
      ]),
    ).toEqual([{ variableId: 'speed', value: 5 }]);
  });
});
