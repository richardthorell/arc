import type { InspectorEntitySnapshot } from './inspectorTypes';

export const waterPresetOverrideBits = {
  windSpeed: 1 << 0,
  windDirection: 1 << 1,
  fetchLength: 1 << 2,
  waveAmplitude: 1 << 3,
  choppiness: 1 << 4,
  foamEnabled: 1 << 5,
  foamThreshold: 1 << 6,
  foamDecay: 1 << 7,
  absorption: 1 << 8,
  scattering: 1 << 9,
  roughness: 1 << 10,
  refractionStrength: 1 << 11,
  quality: 1 << 12,
} as const;

export const waterPresetOverrideForPath: Readonly<Record<string, number>> = {
  'water.windSpeed': waterPresetOverrideBits.windSpeed,
  'water.windDirectionX': waterPresetOverrideBits.windDirection,
  'water.windDirectionY': waterPresetOverrideBits.windDirection,
  'water.fetchLength': waterPresetOverrideBits.fetchLength,
  'water.waveAmplitude': waterPresetOverrideBits.waveAmplitude,
  'water.choppiness': waterPresetOverrideBits.choppiness,
  'water.foamEnabled': waterPresetOverrideBits.foamEnabled,
  'water.foamThreshold': waterPresetOverrideBits.foamThreshold,
  'water.foamDecay': waterPresetOverrideBits.foamDecay,
  'water.absorption': waterPresetOverrideBits.absorption,
  'water.scattering': waterPresetOverrideBits.scattering,
  'water.roughness': waterPresetOverrideBits.roughness,
  'water.refractionStrength': waterPresetOverrideBits.refractionStrength,
  'water.quality': waterPresetOverrideBits.quality,
};

export const waterPresetFieldIsOverridden = (snapshot: InspectorEntitySnapshot, path: string) => {
  const bit = waterPresetOverrideForPath[path];
  return Boolean(bit && snapshot.water && (snapshot.water.presetOverrideMask & bit) !== 0);
};

export const waterPresetInheritanceActive = (snapshot: InspectorEntitySnapshot, path: string) =>
  Boolean(snapshot.water?.presetPath && waterPresetOverrideForPath[path]);
