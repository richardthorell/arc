import type { PropertyComponentSchema } from '../inspector/propertySchema';
import type { SoundAsset } from './soundTypes';

export const soundPropertySchemas: ReadonlyArray<PropertyComponentSchema<SoundAsset>> = [
  {
    id: 'sound-source',
    title: 'Source',
    fields: [
      {
        id: 'source',
        label: 'Audio Source',
        path: 'source',
        type: 'asset',
        assetKind: 'audio',
        assetTypeLabel: 'WAV Audio',
        allowedExtensions: ['.wav'],
        allowEmpty: false,
        referenceMode: 'path',
        tooltip: 'WAV source referenced by this Sound asset.',
      },
    ],
  },
  {
    id: 'sound-playback',
    title: 'Playback',
    fields: [
      {
        id: 'volume',
        label: 'Volume',
        path: 'playback.volume',
        type: 'number',
        precision: 2,
        step: 0.05,
        scrubSensitivity: 0.01,
        min: 0,
      },
      {
        id: 'pitch',
        label: 'Pitch',
        path: 'playback.pitch',
        type: 'number',
        precision: 2,
        step: 0.05,
        scrubSensitivity: 0.01,
        min: 0.01,
      },
      {
        id: 'loop',
        label: 'Loop',
        path: 'playback.loop',
        type: 'boolean',
      },
    ],
  },
  {
    id: 'sound-spatial',
    title: 'Spatial',
    fields: [
      {
        id: 'enabled',
        label: 'Enabled',
        path: 'spatial.enabled',
        type: 'boolean',
      },
      {
        id: 'min-distance',
        label: 'Min Distance',
        path: 'spatial.minDistance',
        type: 'number',
        precision: 2,
        step: 0.25,
        scrubSensitivity: 0.05,
        min: 0,
        unit: 'm',
        visible: (sound) => sound.spatial.enabled,
      },
      {
        id: 'max-distance',
        label: 'Max Distance',
        path: 'spatial.maxDistance',
        type: 'number',
        precision: 2,
        step: 1,
        scrubSensitivity: 0.1,
        min: 0.01,
        unit: 'm',
        visible: (sound) => sound.spatial.enabled,
      },
    ],
  },
];
