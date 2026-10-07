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

  it('groups working Water controls into authoring sections', () => {
    expect(waterSchema).toBeDefined();

    const sectionFor = (id: string) => waterSchema?.fields.find((field) => field.id === id)?.section;
    expect(sectionFor('enabled')).toBe('Body');
    expect(sectionFor('bodyType')).toBe('Body');
    expect(sectionFor('preset')).toBe('Body');
    expect(sectionFor('waterLevel')).toBe('Body');
    expect(sectionFor('followCamera')).toBe('Body');

    expect(sectionFor('windSpeed')).toBe('Waves');
    expect(sectionFor('fetchLength')).toBe('Waves');
    expect(sectionFor('waveAmplitude')).toBe('Waves');
    expect(sectionFor('choppiness')).toBe('Waves');

    expect(sectionFor('material')).toBe('Appearance');
    expect(sectionFor('absorption')).toBe('Appearance');
    expect(sectionFor('scattering')).toBe('Appearance');
    expect(sectionFor('roughness')).toBe('Appearance');
    expect(sectionFor('refractionStrength')).toBe('Appearance');

    expect(sectionFor('visibleDistance')).toBe('Performance');
    expect(sectionFor('quality')).toBe('Performance');
    expect(sectionFor('priority')).toBe('Performance');
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
