export type TexturePreset = 'custom' | 'color' | 'normal_map' | 'data' | 'hdr' | 'ui' | 'environment';
export type TextureSemantic =
  | 'generic_color'
  | 'base_color'
  | 'emissive'
  | 'normal'
  | 'metallic_roughness'
  | 'occlusion'
  | 'clear_coat'
  | 'anisotropy'
  | 'thickness'
  | 'transmission'
  | 'lightmap'
  | 'environment';
export type TextureColorSpace = 'linear' | 'srgb';
export type TextureStreamingMode = 'resident' | 'streamed_mips' | 'virtual_tiles';
export type TextureCompressionPolicy = 'automatic' | 'color' | 'normal' | 'mask' | 'hdr' | 'uncompressed';
export type TexturePowerOfTwoPolicy = 'preserve' | 'resize_down' | 'resize_up';
export type TextureFilterMode = 'nearest' | 'linear';
export type TextureMipFilterMode = 'nearest' | 'linear';
export type TextureAddressMode = 'repeat' | 'clamp_to_edge' | 'mirrored_repeat';
export type TextureMipGenerationFilter = 'nearest' | 'box' | 'bilinear' | 'bicubic' | 'lanczos' | 'kaiser';
export type TextureChannelSource = 'red' | 'green' | 'blue' | 'alpha' | 'zero' | 'one';
export type TextureCurveInterpolation = 'constant' | 'linear' | 'smooth' | 'manual';
export type TextureCurvePoint = {
  x: number;
  y: number;
  inTangent: number;
  outTangent: number;
  interpolation: TextureCurveInterpolation;
};
export type TextureCurve = TextureCurvePoint[];

export type TextureSettingsSnapshot = {
  settingsVersion: number;
  preset: TexturePreset;
  semantic: TextureSemantic;
  colorSpace: TextureColorSpace;
  streamingMode: TextureStreamingMode;
  compression: TextureCompressionPolicy;
  powerOfTwo: TexturePowerOfTwoPolicy;
  minFilter: TextureFilterMode;
  magFilter: TextureFilterMode;
  mipFilter: TextureMipFilterMode;
  wrapU: TextureAddressMode;
  wrapV: TextureAddressMode;
  mipGenerationFilter: TextureMipGenerationFilter;
  maxSize: number;
  anisotropy: number;
  lodBias: number;
  minimumLod: number;
  maximumLod: number;
  alphaCoverageThreshold: number;
  mipSharpen: number;
  ditherMips: boolean;
  debandMips: boolean;
  debandStrength: number;
  brightness: number;
  gamma: number;
  contrast: number;
  saturation: number;
  vibrance: number;
  tintR: number;
  tintG: number;
  tintB: number;
  inputBlack: number;
  inputWhite: number;
  outputBlack: number;
  outputWhite: number;
  curvesEnabled: boolean;
  curveMaster: TextureCurve;
  curveR: TextureCurve;
  curveG: TextureCurve;
  curveB: TextureCurve;
  curveA: TextureCurve;
  channelR: TextureChannelSource;
  channelG: TextureChannelSource;
  channelB: TextureChannelSource;
  channelA: TextureChannelSource;
  invertR: boolean;
  invertG: boolean;
  invertB: boolean;
  invertA: boolean;
  generateMips: boolean;
  preserveAlphaCoverage: boolean;
};

export type TextureSettingsPatch = Partial<Omit<TextureSettingsSnapshot, 'settingsVersion'>>;

type HostResponse<T> = { succeeded: boolean; payload?: T; error?: string };

export async function getTextureSettings(guid: string): Promise<TextureSettingsSnapshot> {
  if (!window.arc?.host?.query) throw new Error('ARC host is unavailable');
  const response = (await window.arc.host.query('texture.settings', { guid })) as HostResponse<TextureSettingsSnapshot>;
  if (!response.succeeded || !response.payload) throw new Error(response.error || 'Could not load texture settings');
  return response.payload;
}

export async function patchTextureSettings(guid: string, patch: TextureSettingsPatch): Promise<void> {
  if (!window.arc?.host?.command) throw new Error('ARC host is unavailable');
  const payload: Record<string, unknown> = { guid, ...patch };
  for (const key of ['curveMaster', 'curveR', 'curveG', 'curveB', 'curveA'] as const) {
    if (patch[key]) payload[key] = JSON.stringify(patch[key]);
  }
  const response = (await window.arc.host.command('texture.settings.patch', payload)) as HostResponse<unknown>;
  if (!response.succeeded) throw new Error(response.error || 'Could not update texture settings');
  window.dispatchEvent(new CustomEvent('arc:texture-settings-changed', { detail: { guid } }));
}
