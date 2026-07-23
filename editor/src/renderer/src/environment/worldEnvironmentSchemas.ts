import type { PropertyComponentSchema } from '../inspector/propertySchema';
import type { HostWorldEnvironment } from './environmentTypes';

export type WorldEnvironmentSectionId =
  | 'general' | 'skySource' | 'sunTime' | 'atmosphere' | 'nightSky'
  | 'clouds' | 'cumulus' | 'cirrus' | 'fog' | 'environmentLighting';

export const worldEnvironmentSchemas: ReadonlyArray<PropertyComponentSchema<HostWorldEnvironment, WorldEnvironmentSectionId>> = [
  {
    id: 'general', title: 'General', fields: [
      { id: 'enabled', label: 'Enabled', path: 'enabled', type: 'boolean' },
      { id: 'skyVisible', label: 'Sky Visible', path: 'skyVisible', type: 'boolean' },
      { id: 'affectLighting', label: 'Affect Lighting', path: 'affectLighting', type: 'boolean' },
    ],
  },
  {
    id: 'skySource', title: 'Sky Source', fields: [
      { id: 'source', label: 'Source', ariaLabel: 'Sky Source', path: 'skySource', type: 'enum', options: [
        { value: 'physicalAtmosphere', label: 'Physical Atmosphere' },
        { value: 'hdri', label: 'HDRI Texture' },
        { value: 'solidColor', label: 'Solid Color' },
      ] },
      { id: 'hdri', label: 'HDRI Texture', path: 'hdriPath', type: 'asset', assetKind: 'texture', allowedExtensions: ['.hdr'], allowEmpty: true,
        visible: (value) => value.skySource === 'hdri' },
      { id: 'solidColor', label: 'Solid Color', path: 'solidColor', type: 'color', precision: 3, min: 0, max: 1, alpha: false,
        visible: (value) => value.skySource === 'solidColor' },
      { id: 'radiance', label: 'Radiance', path: 'radianceIntensity', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0 },
      { id: 'rotation', label: 'Rotation', path: 'hdriRotationDegrees', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, unit: '°',
        visible: (value) => value.skySource === 'hdri' },
    ],
  },
  {
    id: 'sunTime', title: 'Sun & Time', fields: [
      { id: 'sunMode', label: 'Sun Position', path: 'sunMode', type: 'enum', options: [
        { value: 'manualLight', label: 'Manual Light' }, { value: 'geographic', label: 'Geographic' },
      ] },
      { id: 'timeMode', label: 'Clock', path: 'timeMode', type: 'enum', options: [
        { value: 'fixed', label: 'Fixed' }, { value: 'simulated', label: 'Simulated' }, { value: 'systemClock', label: 'System Clock' },
      ], visible: (value) => value.sunMode === 'geographic' },
      { id: 'playing', label: 'Play', path: 'playing', type: 'boolean', visible: (value) => value.sunMode === 'geographic' && value.timeMode === 'simulated' },
      { id: 'localTime', label: 'Time of Day', path: 'localTimeHours', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, unit: 'h', min: 0, max: 23.999,
        visible: (value) => value.sunMode === 'geographic' && value.timeMode !== 'systemClock' },
      { id: 'timeScale', label: 'Time Scale', path: 'timeScale', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, min: 0,
        visible: (value) => value.sunMode === 'geographic' && value.timeMode === 'simulated' },
      { id: 'loopDay', label: 'Loop Day', path: 'loopDay', type: 'boolean', visible: (value) => value.sunMode === 'geographic' && value.timeMode === 'simulated' },
      { id: 'latitude', label: 'Latitude', path: 'latitudeDegrees', type: 'number', precision: 2, step: 0.1, scrubSensitivity: 0.02, unit: '°', min: -90, max: 90,
        visible: (value) => value.sunMode === 'geographic' },
      { id: 'longitude', label: 'Longitude', path: 'longitudeDegrees', type: 'number', precision: 2, step: 0.1, scrubSensitivity: 0.02, unit: '°', min: -180, max: 180,
        visible: (value) => value.sunMode === 'geographic' },
      { id: 'utcOffset', label: 'UTC Offset', path: 'utcOffsetHours', type: 'number', precision: 1, step: 0.5, scrubSensitivity: 0.1, unit: 'h', min: -14, max: 14,
        visible: (value) => value.sunMode === 'geographic' },
      { id: 'northOffset', label: 'North Offset', path: 'northOffsetDegrees', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, unit: '°',
        visible: (value) => value.sunMode === 'geographic' },
      { id: 'year', label: 'Year', path: 'year', type: 'number', precision: 0, step: 1, scrubSensitivity: 0.2, min: 1, max: 9999,
        visible: (value) => value.sunMode === 'geographic' && value.timeMode !== 'systemClock' },
      { id: 'month', label: 'Month', path: 'month', type: 'number', precision: 0, step: 1, scrubSensitivity: 0.1, min: 1, max: 12,
        visible: (value) => value.sunMode === 'geographic' && value.timeMode !== 'systemClock' },
      { id: 'day', label: 'Day', path: 'day', type: 'number', precision: 0, step: 1, scrubSensitivity: 0.1, min: 1, max: 31,
        visible: (value) => value.sunMode === 'geographic' && value.timeMode !== 'systemClock' },
      { id: 'automaticSun', label: 'Automatic Sun', path: 'automaticSunLight', type: 'boolean' },
      { id: 'sunIntensity', label: 'Sun Intensity', path: 'sunIntensityMultiplier', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0 },
      { id: 'sunTemperature', label: 'Sun Temperature', path: 'sunTemperatureMultiplier', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0.1 },
    ],
  },
  {
    id: 'atmosphere', title: 'Atmosphere', badge: 'Advanced', collapsedByDefault: true,
    visible: (value) => value.skySource === 'physicalAtmosphere', fields: [
      { id: 'tint', label: 'Atmosphere Tint', path: 'atmosphereTint', type: 'color', precision: 3, min: 0, max: 1, alpha: false },
      { id: 'ground', label: 'Ground Albedo', path: 'groundAlbedo', type: 'color', precision: 3, min: 0, max: 1, alpha: false },
      { id: 'rayleigh', label: 'Rayleigh', path: 'rayleighStrength', type: 'number', precision: 3, step: 0.02, scrubSensitivity: 0.005, min: 0 },
      { id: 'mie', label: 'Mie / Haze', path: 'mieStrength', type: 'number', precision: 3, step: 0.02, scrubSensitivity: 0.005, min: 0 },
      { id: 'ozone', label: 'Ozone', path: 'ozoneStrength', type: 'number', precision: 3, step: 0.02, scrubSensitivity: 0.005, min: 0 },
      { id: 'anisotropy', label: 'Mie Anisotropy', path: 'mieAnisotropy', type: 'number', precision: 3, step: 0.01, scrubSensitivity: 0.002, min: -0.98, max: 0.98 },
      { id: 'exposure', label: 'Exposure', path: 'exposure', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0 },
      { id: 'rayleighHeight', label: 'Rayleigh Height', path: 'rayleighScaleHeight', type: 'number', precision: 2, step: 0.1, scrubSensitivity: 0.02, unit: 'km', min: 0.01 },
      { id: 'mieHeight', label: 'Mie Height', path: 'mieScaleHeight', type: 'number', precision: 2, step: 0.1, scrubSensitivity: 0.02, unit: 'km', min: 0.01 },
      { id: 'multiScattering', label: 'Multi Scattering', path: 'multiScatteringFactor', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0 },
      { id: 'sunDiskSize', label: 'Sun Disk Size', path: 'sunDiskSize', type: 'number', precision: 3, step: 0.001, scrubSensitivity: 0.0002, min: 0 },
      { id: 'sunDiskPower', label: 'Sun Disk Power', path: 'sunDiskIntensity', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0 },
      { id: 'planetRadius', label: 'Planet Radius', path: 'planetRadius', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, unit: 'km', min: 1 },
      { id: 'atmosphereRadius', label: 'Atmosphere Radius', path: 'atmosphereRadius', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, unit: 'km', min: 1 },
    ],
  },
  {
    id: 'nightSky', title: 'Night Sky', fields: [
      { id: 'moon', label: 'Moon', path: 'moonEnabled', type: 'boolean' },
      { id: 'automaticPhase', label: 'Automatic Phase', path: 'automaticMoonPhase', type: 'boolean', visible: (value) => value.moonEnabled },
      { id: 'moonPhase', label: 'Moon Phase', path: 'moonPhase', type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1,
        visible: (value) => value.moonEnabled && !value.automaticMoonPhase },
      { id: 'moonBrightness', label: 'Moon Brightness', path: 'moonIntensity', type: 'number', precision: 2, step: 0.02, scrubSensitivity: 0.005, min: 0, visible: (value) => value.moonEnabled },
      { id: 'moonRadius', label: 'Angular Radius', path: 'moonAngularRadiusDegrees', type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, unit: '°', min: 0.01, visible: (value) => value.moonEnabled },
      { id: 'stars', label: 'Stars', path: 'starsEnabled', type: 'boolean' },
      { id: 'starDensity', label: 'Star Density', path: 'starDensity', type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: (value) => value.starsEnabled },
      { id: 'starIntensity', label: 'Star Intensity', path: 'starIntensity', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0, visible: (value) => value.starsEnabled },
      { id: 'starTwinkle', label: 'Star Twinkle', path: 'starTwinkle', type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: (value) => value.starsEnabled },
    ],
  },
  {
    id: 'clouds', title: 'Clouds', fields: [
      { id: 'enabled', label: 'Enabled', path: 'cloudsEnabled', type: 'boolean' },
      { id: 'shadows', label: 'Cloud Shadows', path: 'cloudShadows', type: 'boolean', visible: (value) => value.cloudsEnabled },
    ],
  },
  {
    id: 'cumulus', title: 'Cumulus Cloud Layer', visible: (value) => value.cloudsEnabled, fields: cloudLayerFields('cumulus'),
  },
  {
    id: 'cirrus', title: 'Cirrus Cloud Layer', visible: (value) => value.cloudsEnabled, fields: cloudLayerFields('cirrus'),
  },
  {
    id: 'fog', title: 'Fog', fields: [
      { id: 'enabled', label: 'Enabled', path: 'fogEnabled', type: 'boolean' },
      { id: 'color', label: 'Fog Color', path: 'fogColor', type: 'color', precision: 3, min: 0, max: 1, alpha: false, visible: (value) => value.fogEnabled },
      { id: 'density', label: 'Density', path: 'fogDensity', type: 'number', precision: 4, step: 0.001, scrubSensitivity: 0.0002, min: 0, visible: (value) => value.fogEnabled },
      { id: 'falloff', label: 'Height Falloff', path: 'fogHeightFalloff', type: 'number', precision: 3, step: 0.01, scrubSensitivity: 0.002, min: 0, visible: (value) => value.fogEnabled },
      { id: 'start', label: 'Start Distance', path: 'fogStartDistance', type: 'number', precision: 1, step: 1, scrubSensitivity: 0.2, unit: 'm', min: 0, visible: (value) => value.fogEnabled },
      { id: 'opacity', label: 'Max Opacity', path: 'fogMaxOpacity', type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: (value) => value.fogEnabled },
      { id: 'sunScattering', label: 'Sun Scattering', path: 'fogSunScattering', type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, visible: (value) => value.fogEnabled },
    ],
  },
  {
    id: 'environmentLighting', title: 'Environment Lighting', fields: [
      { id: 'enabled', label: 'Enabled', path: 'lightingEnabled', type: 'boolean' },
      { id: 'source', label: 'Source', ariaLabel: 'Environment Lighting Source', path: 'lightingSource', type: 'enum', options: [
        { value: 'followSky', label: 'Follow Sky' }, { value: 'hdri', label: 'HDRI Texture' }, { value: 'constantColor', label: 'Constant Color' },
      ], visible: (value) => value.lightingEnabled },
      { id: 'hdri', label: 'HDRI Texture', path: 'hdriPath', type: 'asset', assetKind: 'texture', allowedExtensions: ['.hdr'], allowEmpty: true,
        visible: (value) => value.lightingEnabled && value.lightingSource === 'hdri' },
      { id: 'color', label: 'Ambient Color', path: 'lightingColor', type: 'color', precision: 3, min: 0, max: 1, alpha: false,
        visible: (value) => value.lightingEnabled && value.lightingSource === 'constantColor' },
      { id: 'diffuse', label: 'Diffuse', path: 'diffuseIntensity', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0, visible: (value) => value.lightingEnabled },
      { id: 'specular', label: 'Specular', path: 'specularIntensity', type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0, visible: (value) => value.lightingEnabled },
    ],
  },
];

function cloudLayerFields(layer: 'cumulus' | 'cirrus'): PropertyComponentSchema<HostWorldEnvironment>['fields'] {
  const path = (field: string) => `${layer}.${field}`;
  const active = (value: HostWorldEnvironment) => value[layer].enabled;
  return [
    { id: 'enabled', label: 'Enabled', path: path('enabled'), type: 'boolean' },
    { id: 'coverage', label: 'Coverage', path: path('coverage'), type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: active },
    { id: 'density', label: 'Density', path: path('density'), type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: active },
    { id: 'altitude', label: 'Altitude', path: path('altitude'), type: 'number', precision: 0, step: 50, scrubSensitivity: 5, unit: 'm', min: 0, visible: active },
    { id: 'thickness', label: 'Thickness', path: path('thickness'), type: 'number', precision: 0, step: 10, scrubSensitivity: 2, unit: 'm', min: 0, visible: active },
    { id: 'scale', label: 'Scale', path: path('scale'), type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0.001, visible: active },
    { id: 'detail', label: 'Detail', path: path('detail'), type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: active },
    { id: 'softness', label: 'Softness', path: path('softness'), type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: active },
    { id: 'windX', label: 'Wind X', path: path('windX'), type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: -1, max: 1, visible: active },
    { id: 'windY', label: 'Wind Y', path: path('windY'), type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: -1, max: 1, visible: active },
    { id: 'windSpeed', label: 'Wind Speed', path: path('windSpeed'), type: 'number', precision: 1, step: 0.5, scrubSensitivity: 0.1, unit: 'm/s', min: 0, visible: active },
    { id: 'lighting', label: 'Lighting', path: path('lightingStrength'), type: 'number', precision: 2, step: 0.05, scrubSensitivity: 0.01, min: 0, visible: active },
    { id: 'silverLining', label: 'Silver Lining', path: path('silverLining'), type: 'number', precision: 2, step: 0.01, scrubSensitivity: 0.002, min: 0, max: 1, visible: active },
  ];
}
