import { describe, expect, it } from 'vitest';

import { soundPropertySchemas } from './soundSchemas';

describe('Sound property schemas', () => {
  it('represent every editable .arcsound v1 property through data-driven fields', () => {
    const fields = soundPropertySchemas.flatMap((schema) => schema.fields);
    expect(fields.map((field) => field.path)).toEqual([
      'source',
      'playback.volume',
      'playback.pitch',
      'playback.loop',
      'spatial.enabled',
      'spatial.minDistance',
      'spatial.maxDistance',
    ]);

    expect(fields.find((field) => field.path === 'source')).toMatchObject({
      type: 'asset',
      assetKind: 'audio',
      allowedExtensions: ['.wav'],
    });
    expect(fields.find((field) => field.path === 'playback.volume')).toMatchObject({ type: 'number', min: 0 });
    expect(fields.find((field) => field.path === 'playback.pitch')).toMatchObject({ type: 'number', min: 0.01 });
    expect(fields.find((field) => field.path === 'playback.loop')).toMatchObject({ type: 'boolean' });
    expect(fields.find((field) => field.path === 'spatial.enabled')).toMatchObject({ type: 'boolean' });
  });

  it('hides attenuation controls when spatial playback is disabled', () => {
    const spatial = soundPropertySchemas.find((schema) => schema.id === 'sound-spatial')!;
    const context = {
      kind: 'sound' as const,
      version: 1 as const,
      source: 'Audio/Test.wav',
      playback: { loop: false, volume: 1, pitch: 1 },
      spatial: { enabled: false, minDistance: 1, maxDistance: 25 },
    };

    expect(spatial.fields.find((field) => field.path === 'spatial.minDistance')?.visible?.(context)).toBe(false);
    expect(spatial.fields.find((field) => field.path === 'spatial.maxDistance')?.visible?.(context)).toBe(false);
  });
});
