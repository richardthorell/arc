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
export type TextureMipGenerationFilter = 'box' | 'nearest';

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
  const response = (await window.arc.host.command('texture.settings.patch', {
    guid,
    ...patch,
  })) as HostResponse<unknown>;
  if (!response.succeeded) throw new Error(response.error || 'Could not update texture settings');
}
