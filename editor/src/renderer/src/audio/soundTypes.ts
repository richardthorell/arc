export type SoundAsset = {
  kind: 'sound';
  version: 1;
  source: string;
  playback: {
    loop: boolean;
    volume: number;
    pitch: number;
  };
  spatial: {
    enabled: boolean;
    minDistance: number;
    maxDistance: number;
  };
};

export const createDefaultSoundAsset = (): SoundAsset => ({
  kind: 'sound',
  version: 1,
  source: '',
  playback: {
    loop: false,
    volume: 1,
    pitch: 1,
  },
  spatial: {
    enabled: true,
    minDistance: 1,
    maxDistance: 25,
  },
});

export const parseSoundAsset = (value: unknown): SoundAsset => {
  if (!value || typeof value !== 'object') throw new Error('Sound document must be an object');
  const candidate = value as Partial<SoundAsset>;
  if (candidate.kind !== 'sound' || candidate.version !== 1) throw new Error('Sound must use .arcsound schema v1');
  if (typeof candidate.source !== 'string') throw new Error('Sound requires a WAV source path');

  const playback = candidate.playback;
  if (
    !playback ||
    typeof playback.loop !== 'boolean' ||
    typeof playback.volume !== 'number' ||
    !Number.isFinite(playback.volume) ||
    playback.volume < 0 ||
    typeof playback.pitch !== 'number' ||
    !Number.isFinite(playback.pitch) ||
    playback.pitch <= 0
  )
    throw new Error('Sound playback settings are invalid');

  const spatial = candidate.spatial;
  if (
    !spatial ||
    typeof spatial.enabled !== 'boolean' ||
    typeof spatial.minDistance !== 'number' ||
    !Number.isFinite(spatial.minDistance) ||
    spatial.minDistance < 0 ||
    typeof spatial.maxDistance !== 'number' ||
    !Number.isFinite(spatial.maxDistance) ||
    spatial.maxDistance <= spatial.minDistance
  )
    throw new Error('Sound spatial settings are invalid');

  return {
    kind: 'sound',
    version: 1,
    source: candidate.source,
    playback: {
      loop: playback.loop,
      volume: playback.volume,
      pitch: playback.pitch,
    },
    spatial: {
      enabled: spatial.enabled,
      minDistance: spatial.minDistance,
      maxDistance: spatial.maxDistance,
    },
  };
};
