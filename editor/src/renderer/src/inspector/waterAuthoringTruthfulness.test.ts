import { describe, expect, it } from 'vitest';

import { inspectorComponentSchemas } from './componentSchemas';
import type { InspectorEntitySnapshot } from './inspectorTypes';

const waterSchema = inspectorComponentSchemas.find((schema) => schema.id === 'water');

const snapshot = (bodyType: 'ocean' | 'lake' | 'river') =>
  ({
    water: {
      bodyType,
    },
  }) as InspectorEntitySnapshot;

describe('Water inspector truthfulness', () => {
  it('does not expose later-milestone Water controls as if they were functional', () => {
    expect(waterSchema).toBeDefined();
    const ids = new Set(waterSchema?.fields.map((field) => field.id));

    for (const id of [
      'foamEnabled',
      'foamThreshold',
      'foamDecay',
      'shorelineEnabled',
      'shorelineFoamWidth',
      'shallowWaveDampingDistance',
      'runupDistance',
      'underwaterEnabled',
      'causticsEnabled',
      'queriesEnabled',
      'buoyancyEnabled',
    ])
      expect(ids.has(id)).toBe(false);
  });

  it('shows spectral Ocean controls only for the Ocean provider', () => {
    expect(waterSchema).toBeDefined();
    for (const id of [
      'followCamera',
      'windSpeed',
      'windDirectionX',
      'windDirectionY',
      'fetchLength',
      'waveAmplitude',
      'choppiness',
      'quality',
    ]) {
      const field = waterSchema?.fields.find((candidate) => candidate.id === id);
      expect(field?.visible).toBeDefined();
      expect(field?.visible?.(snapshot('ocean'))).toBe(true);
      expect(field?.visible?.(snapshot('lake'))).toBe(false);
      expect(field?.visible?.(snapshot('river'))).toBe(false);
    }
  });
});
