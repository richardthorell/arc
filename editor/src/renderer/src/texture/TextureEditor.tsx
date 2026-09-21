import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, UIEvent, WheelEvent } from 'react';
import { Image, Maximize2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { UiButton, UiNumericInput, UiPanel, UiPropertyCard, UiSelect, UiToggleButton } from '../ui';
import { setTextureEditorViewState, useTextureEditorViewState } from './textureEditorViewState';
import { TextureStage3Controls } from './TextureStage3Controls';
import { TextureCurveControls } from './TextureCurveControls';
import { analyzeTexturePreview, type TexturePreviewAnalysis } from './texturePreviewProcessing';
import { useTextureSettings } from './useTextureSettings';
import {
  getTextureSettings,
  patchTextureSettings,
  type TextureAddressMode,
  type TextureColorSpace,
  type TextureCompressionPolicy,
  type TextureFilterMode,
  type TextureMipFilterMode,
  type TextureMipGenerationFilter,
  type TextureMipPolicy,
  type TexturePowerOfTwoPolicy,
  type TexturePreset,
  type TextureSemantic,
  type TextureSettingsSnapshot,
  type TextureStreamingMode,
} from './textureSettings';

import '../inspector/inspector.css';
import './textureEditor.css';

type HostResponse<T = unknown> = {
  succeeded: boolean;
  payload?: T;
};

type HostAssetThumbnailSnapshot = {
  path: string;
  width: number;
  height: number;
  dataUrl: string;
};

type RulerMark = {
  value: number;
  position: number;
  major: boolean;
};

type ViewportMetrics = {
  scrollLeft: number;
  scrollTop: number;
  width: number;
  height: number;
};

type PanState = {
  pointerId: number;
  startX: number;
  startY: number;
  scrollLeft: number;
  scrollTop: number;
};

const minZoom = 0.25;
const maxZoom = 16;
const previewPadding = 28;
const defaultInspectorWidth = 400;
const minInspectorWidth = 320;
const maxInspectorWidth = 680;

const extensionOf = (path: string) => {
  const fileName = path.replaceAll('\\', '/').split('/').at(-1) ?? path;
  const dot = fileName.lastIndexOf('.');
  return dot > 0 && dot < fileName.length - 1 ? fileName.slice(dot + 1).toLocaleUpperCase() : 'Unknown';
};

const formatBytes = (bytes: number | undefined) => {
  if (bytes === undefined || !Number.isFinite(bytes) || bytes < 0) return 'Not reported';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${units[unit]}`;
};

const dimensionsOf = (asset: AssetItem) => {
  if (asset.width === undefined || asset.height === undefined) return 'Not reported';
  return `${asset.width} × ${asset.height}${asset.depth && asset.depth > 1 ? ` × ${asset.depth}` : ''}`;
};

const textureTypeOf = (asset: AssetItem) => (asset.depth && asset.depth > 1 ? '3D Texture' : '2D Texture');
const clampZoom = (value: number) => Math.min(maxZoom, Math.max(minZoom, value));

const rulerInterval = (zoom: number) => {
  const candidates = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
  return candidates.find((candidate) => candidate * zoom >= 42) ?? candidates.at(-1)!;
};

const rulerMarks = (size: number, zoom: number): RulerMark[] => {
  const majorInterval = rulerInterval(zoom);
  const minorInterval = majorInterval / 5;
  const count = Math.ceil(size / minorInterval);
  return Array.from({ length: count + 1 }, (_, index) => {
    const value = Math.min(size, index * minorInterval);
    return {
      value,
      position: value * zoom,
      major: index % 5 === 0,
    };
  });
};

function TextureValue({ value }: { value: string }) {
  return (
    <span className="texture-inspector-value" title={value}>
      {value}
    </span>
  );
}

function TextureSelect({
  ariaLabel,
  value,
  options,
  disabled,
  onChange,
}: {
  ariaLabel: string;
  value: string;
  options: ReadonlyArray<{ value: string; label: string; disabled?: boolean }>;
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <UiSelect ariaLabel={ariaLabel} disabled={disabled} onValueChange={onChange} options={options} value={value} />
  );
}

function TextureNumber({
  ariaLabel,
  value,
  min,
  max,
  step,
  precision,
  disabled,
  onChange,
}: {
  ariaLabel: string;
  value: number;
  min?: number;
  max?: number;
  step: number;
  precision: number;
  disabled?: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <UiNumericInput
      ariaLabel={ariaLabel}
      disabled={disabled}
      max={max}
      min={min}
      onCommit={onChange}
      precision={precision}
      scrubSensitivity={step}
      step={step}
      value={value}
    />
  );
}

function TextureToggle({
  ariaLabel,
  checked,
  disabled,
  onChange,
}: {
  ariaLabel: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return <UiToggleButton aria-label={ariaLabel} checked={checked} disabled={disabled} onCheckedChange={onChange} />;
}

const texturePresetOptions = [
  { value: 'custom', label: 'Custom' },
  { value: 'color', label: 'Color' },
  { value: 'normal_map', label: 'Normal Map' },
  { value: 'data', label: 'Data / Mask' },
  { value: 'hdr', label: 'HDR' },
  { value: 'ui', label: 'UI' },
  { value: 'environment', label: 'Environment' },
];

const textureSemanticOptions = [
  { value: 'generic_color', label: 'Generic Color' },
  { value: 'base_color', label: 'Base Color' },
  { value: 'emissive', label: 'Emissive' },
  { value: 'normal', label: 'Normal' },
  { value: 'metallic_roughness', label: 'Metallic / Roughness' },
  { value: 'occlusion', label: 'Occlusion' },
  { value: 'clear_coat', label: 'Clear Coat' },
  { value: 'anisotropy', label: 'Anisotropy' },
  { value: 'thickness', label: 'Thickness' },
  { value: 'transmission', label: 'Transmission' },
  { value: 'lightmap', label: 'Lightmap' },
  { value: 'environment', label: 'Environment' },
];

const textureAddressOptions = [
  { value: 'repeat', label: 'Repeat' },
  { value: 'clamp_to_edge', label: 'Clamp' },
  { value: 'mirrored_repeat', label: 'Mirror' },
];

const textureFilterOptions = [
  { value: 'linear', label: 'Linear' },
  { value: 'nearest', label: 'Nearest' },
];

function TextureInspector({ asset, histogram }: { asset: AssetItem; histogram?: TexturePreviewAnalysis['histogram'] }) {
  const ddsSource = extensionOf(asset.path) === 'DDS';
  const [settings, setSettings] = useState<TextureSettingsSnapshot | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({
    texture: false,
    sampling: false,
    mipmaps: false,
    compression: true,
    streaming: true,
    import: true,
    asset: true,
  });
  const toggleSection = (section: string) =>
    setCollapsedSections((current) => ({ ...current, [section]: !current[section] }));

  useEffect(() => {
    let active = true;
    if (!asset.guid || asset.readOnly) {
      setSettings(null);
      setSettingsError(asset.readOnly ? 'Read-only texture' : 'Settings unavailable');
      return;
    }
    void getTextureSettings(asset.guid)
      .then((value) => {
        if (active) {
          setSettings(value);
          setSettingsError(null);
        }
      })
      .catch((error: unknown) => {
        if (active) setSettingsError(error instanceof Error ? error.message : 'Could not load texture settings');
      });
    return () => {
      active = false;
    };
  }, [asset.guid, asset.generation, asset.readOnly]);

  const updateSettings = async (patch: Parameters<typeof patchTextureSettings>[1]) => {
    if (!asset.guid || !settings || settingsBusy) return;
    const previous = settings;
    const optimistic = { ...settings, ...patch };
    setSettings(optimistic);
    setSettingsBusy(true);
    setSettingsError(null);
    try {
      await patchTextureSettings(asset.guid, patch);
      setSettings(await getTextureSettings(asset.guid));
    } catch (error) {
      setSettings(previous);
      setSettingsError(error instanceof Error ? error.message : 'Could not update texture settings');
    } finally {
      setSettingsBusy(false);
    }
  };

  const mipProcessingDisabled =
    !settings ||
    settingsBusy ||
    settings.mipPolicy === 'none' ||
    (ddsSource && settings.mipPolicy === 'preserve_source');

  return (
    <UiPanel aria-label="Texture details" className="texture-inspector" role="complementary" variant="inspector">
      <div className="texture-inspector-sections">
        <UiPropertyCard
          className="texture-inspector-section"
          collapsed={collapsedSections.texture}
          fields={[
            { id: 'type', label: 'Type', control: <TextureValue value={textureTypeOf(asset)} /> },
            { id: 'dimensions', label: 'Dimensions', control: <TextureValue value={dimensionsOf(asset)} /> },
            {
              id: 'depth',
              label: 'Depth / Layers',
              control: <TextureValue value={asset.depth === undefined ? '1' : String(asset.depth)} />,
            },
            {
              id: 'format',
              label: 'Format',
              control: <TextureValue value={asset.textureFormat ?? extensionOf(asset.path)} />,
            },
            ...(settings
              ? [
                  {
                    id: 'preset',
                    label: 'Preset',
                    control: (
                      <TextureSelect
                        ariaLabel="Texture preset"
                        disabled={settingsBusy}
                        options={texturePresetOptions}
                        value={settings.preset}
                        onChange={(preset) => void updateSettings({ preset: preset as TexturePreset })}
                      />
                    ),
                  },
                  {
                    id: 'semantic',
                    label: 'Semantic',
                    control: (
                      <TextureSelect
                        ariaLabel="Texture semantic"
                        disabled={settingsBusy}
                        options={textureSemanticOptions}
                        value={settings.semantic}
                        onChange={(semantic) => void updateSettings({ semantic: semantic as TextureSemantic })}
                      />
                    ),
                  },
                  {
                    id: 'color-space',
                    label: 'Color Space',
                    control: (
                      <TextureSelect
                        ariaLabel="Texture color space"
                        disabled={settingsBusy}
                        options={[
                          { value: 'srgb', label: 'sRGB' },
                          { value: 'linear', label: 'Linear' },
                        ]}
                        value={settings.colorSpace}
                        onChange={(colorSpace) => void updateSettings({ colorSpace: colorSpace as TextureColorSpace })}
                      />
                    ),
                  },
                ]
              : [
                  {
                    id: 'color-space',
                    label: 'Color Space',
                    control: <TextureValue value={settingsError ?? 'Loading…'} />,
                  },
                ]),
            { id: 'alpha', label: 'Alpha', control: <TextureValue value="Not reported" /> },
            {
              id: 'source-size',
              label: 'Source Size',
              control: <TextureValue value={formatBytes(asset.sourceBytes)} />,
            },
          ]}
          onToggle={() => toggleSection('texture')}
          title="Texture"
        />

        <TextureStage3Controls asset={asset} />
        <TextureCurveControls asset={asset} histogram={histogram} />

        <UiPropertyCard
          className="texture-inspector-section"
          collapsed={collapsedSections.sampling}
          fields={
            settings
              ? [
                  {
                    id: 'wrap-u',
                    label: 'Wrap U',
                    control: (
                      <TextureSelect
                        ariaLabel="Wrap U"
                        disabled={settingsBusy}
                        options={textureAddressOptions}
                        value={settings.wrapU}
                        onChange={(wrapU) => void updateSettings({ wrapU: wrapU as TextureAddressMode })}
                      />
                    ),
                  },
                  {
                    id: 'wrap-v',
                    label: 'Wrap V',
                    control: (
                      <TextureSelect
                        ariaLabel="Wrap V"
                        disabled={settingsBusy}
                        options={textureAddressOptions}
                        value={settings.wrapV}
                        onChange={(wrapV) => void updateSettings({ wrapV: wrapV as TextureAddressMode })}
                      />
                    ),
                  },
                  {
                    id: 'min-filter',
                    label: 'Min Filter',
                    control: (
                      <TextureSelect
                        ariaLabel="Min Filter"
                        disabled={settingsBusy}
                        options={textureFilterOptions}
                        value={settings.minFilter}
                        onChange={(minFilter) => void updateSettings({ minFilter: minFilter as TextureFilterMode })}
                      />
                    ),
                  },
                  {
                    id: 'mag-filter',
                    label: 'Mag Filter',
                    control: (
                      <TextureSelect
                        ariaLabel="Mag Filter"
                        disabled={settingsBusy}
                        options={textureFilterOptions}
                        value={settings.magFilter}
                        onChange={(magFilter) => void updateSettings({ magFilter: magFilter as TextureFilterMode })}
                      />
                    ),
                  },
                  {
                    id: 'mip-filter',
                    label: 'Mip Filter',
                    control: (
                      <TextureSelect
                        ariaLabel="Mip Filter"
                        disabled={settingsBusy}
                        options={textureFilterOptions}
                        value={settings.mipFilter}
                        onChange={(mipFilter) => void updateSettings({ mipFilter: mipFilter as TextureMipFilterMode })}
                      />
                    ),
                  },
                  {
                    id: 'anisotropy',
                    label: 'Anisotropy',
                    control: (
                      <TextureNumber
                        ariaLabel="Anisotropy"
                        disabled={settingsBusy}
                        max={16}
                        min={1}
                        onChange={(anisotropy) => void updateSettings({ anisotropy })}
                        precision={0}
                        step={1}
                        value={settings.anisotropy}
                      />
                    ),
                  },
                ]
              : [
                  {
                    id: 'sampling-unavailable',
                    label: 'Sampling',
                    control: <TextureValue value={settingsError ?? 'Loading…'} />,
                  },
                ]
          }
          onToggle={() => toggleSection('sampling')}
          title="Sampling"
        />

        <UiPropertyCard
          className="texture-inspector-section"
          collapsed={collapsedSections.mipmaps}
          fields={[
            ...(settings
              ? [
                  {
                    id: 'source-mips',
                    label: 'Source Mips',
                    control: (
                      <TextureValue value={asset.mipLevels === undefined ? 'Not reported' : String(asset.mipLevels)} />
                    ),
                  },
                  {
                    id: 'mip-policy',
                    label: 'Mip Policy',
                    control: (
                      <TextureSelect
                        ariaLabel="Texture mip policy"
                        disabled={settingsBusy}
                        options={[
                          { value: 'preserve_source', label: 'Preserve Source' },
                          { value: 'generate', label: 'Generate', disabled: ddsSource },
                          {
                            value: 'none',
                            label: 'None',
                            disabled: settings.streamingMode !== 'resident',
                          },
                        ]}
                        value={settings.mipPolicy}
                        onChange={(mipPolicy) => void updateSettings({ mipPolicy: mipPolicy as TextureMipPolicy })}
                      />
                    ),
                  },
                  {
                    id: 'mip-source',
                    label: 'Mip Source',
                    control: (
                      <TextureValue
                        value={
                          settings.mipPolicy === 'generate'
                            ? 'Generated'
                            : settings.mipPolicy === 'none'
                              ? 'Base level only'
                              : ddsSource
                                ? 'Authored / preserved'
                                : 'Generate if absent'
                        }
                      />
                    ),
                  },
                  {
                    id: 'generation-filter',
                    label: 'Generation Filter',
                    control: (
                      <TextureSelect
                        ariaLabel="Generation Filter"
                        disabled={settingsBusy}
                        options={[
                          { value: 'kaiser', label: 'Kaiser' },
                          { value: 'lanczos', label: 'Lanczos' },
                          { value: 'bicubic', label: 'Bicubic' },
                          { value: 'bilinear', label: 'Bilinear' },
                          { value: 'box', label: 'Box' },
                          { value: 'nearest', label: 'Nearest' },
                        ]}
                        value={settings.mipGenerationFilter}
                        onChange={(mipGenerationFilter) =>
                          void updateSettings({
                            mipGenerationFilter: mipGenerationFilter as TextureMipGenerationFilter,
                          })
                        }
                      />
                    ),
                  },
                  {
                    id: 'sharpen',
                    label: 'Sharpen',
                    control: (
                      <TextureNumber
                        ariaLabel="Sharpen"
                        disabled={mipProcessingDisabled}
                        max={2}
                        min={0}
                        onChange={(mipSharpen) => void updateSettings({ mipSharpen })}
                        precision={2}
                        step={0.05}
                        value={settings.mipSharpen}
                      />
                    ),
                  },
                  {
                    id: 'dither',
                    label: 'Dither',
                    control: (
                      <TextureToggle
                        ariaLabel="Dither"
                        checked={settings.ditherMips}
                        disabled={mipProcessingDisabled}
                        onChange={(ditherMips) => void updateSettings({ ditherMips })}
                      />
                    ),
                  },
                  {
                    id: 'deband',
                    label: 'De-band',
                    control: (
                      <TextureToggle
                        ariaLabel="De-band"
                        checked={settings.debandMips}
                        disabled={mipProcessingDisabled}
                        onChange={(debandMips) => void updateSettings({ debandMips })}
                      />
                    ),
                  },
                  {
                    id: 'deband-strength',
                    label: 'De-band Strength',
                    control: (
                      <TextureNumber
                        ariaLabel="De-band Strength"
                        disabled={mipProcessingDisabled || !settings.debandMips}
                        max={1}
                        min={0}
                        onChange={(debandStrength) => void updateSettings({ debandStrength })}
                        precision={2}
                        step={0.05}
                        value={settings.debandStrength}
                      />
                    ),
                  },
                  {
                    id: 'preserve-alpha',
                    label: 'Preserve Alpha',
                    control: (
                      <TextureToggle
                        ariaLabel="Preserve Alpha"
                        checked={settings.preserveAlphaCoverage}
                        disabled={settingsBusy}
                        onChange={(preserveAlphaCoverage) => void updateSettings({ preserveAlphaCoverage })}
                      />
                    ),
                  },
                  {
                    id: 'alpha-threshold',
                    label: 'Alpha Threshold',
                    control: (
                      <TextureNumber
                        ariaLabel="Alpha Threshold"
                        disabled={settingsBusy || !settings.preserveAlphaCoverage}
                        max={1}
                        min={0}
                        onChange={(alphaCoverageThreshold) => void updateSettings({ alphaCoverageThreshold })}
                        precision={2}
                        step={0.05}
                        value={settings.alphaCoverageThreshold}
                      />
                    ),
                  },
                  {
                    id: 'lod-bias',
                    label: 'LOD Bias',
                    control: (
                      <TextureNumber
                        ariaLabel="LOD Bias"
                        disabled={settingsBusy}
                        onChange={(lodBias) => void updateSettings({ lodBias })}
                        precision={2}
                        step={0.25}
                        value={settings.lodBias}
                      />
                    ),
                  },
                  {
                    id: 'min-lod',
                    label: 'Min LOD',
                    control: (
                      <TextureNumber
                        ariaLabel="Min LOD"
                        disabled={settingsBusy}
                        min={0}
                        onChange={(minimumLod) => void updateSettings({ minimumLod })}
                        precision={2}
                        step={0.25}
                        value={settings.minimumLod}
                      />
                    ),
                  },
                  {
                    id: 'max-lod',
                    label: 'Max LOD',
                    control: (
                      <TextureNumber
                        ariaLabel="Max LOD"
                        disabled={settingsBusy}
                        min={0}
                        onChange={(maximumLod) => void updateSettings({ maximumLod })}
                        precision={2}
                        step={0.25}
                        value={settings.maximumLod}
                      />
                    ),
                  },
                ]
              : []),
            {
              id: 'mip-count',
              label: 'Mip Count',
              control: (
                <TextureValue value={asset.mipLevels === undefined ? 'Not reported' : String(asset.mipLevels)} />
              ),
            },
          ]}
          onToggle={() => toggleSection('mipmaps')}
          title="Mipmaps"
        />

        <UiPropertyCard
          className="texture-inspector-section"
          collapsed={collapsedSections.compression}
          fields={[
            ...(settings
              ? [
                  {
                    id: 'policy',
                    label: 'Policy',
                    control: (
                      <TextureSelect
                        ariaLabel="Compression Policy"
                        disabled={settingsBusy}
                        options={[
                          { value: 'automatic', label: 'Automatic' },
                          { value: 'color', label: 'Color' },
                          { value: 'normal', label: 'Normal' },
                          { value: 'mask', label: 'Mask' },
                          { value: 'hdr', label: 'HDR' },
                          { value: 'uncompressed', label: 'Uncompressed' },
                        ]}
                        value={settings.compression}
                        onChange={(compression) =>
                          void updateSettings({ compression: compression as TextureCompressionPolicy })
                        }
                      />
                    ),
                  },
                ]
              : []),
            {
              id: 'gpu-format',
              label: 'GPU Format',
              control: <TextureValue value={asset.textureFormat ?? 'Resolved at cook'} />,
            },
            {
              id: 'artifact-size',
              label: 'Artifact Size',
              control: <TextureValue value={formatBytes(asset.artifactSize)} />,
            },
          ]}
          onToggle={() => toggleSection('compression')}
          title="Compression"
        />

        <UiPropertyCard
          className="texture-inspector-section"
          collapsed={collapsedSections.streaming}
          fields={[
            {
              id: 'mode',
              label: 'Mode',
              control: settings ? (
                <TextureSelect
                  ariaLabel="Texture streaming mode"
                  disabled={settingsBusy}
                  options={[
                    { value: 'resident', label: 'Resident' },
                    { value: 'streamed_mips', label: 'Streamed Mips' },
                    { value: 'virtual_tiles', label: 'Virtual Tiles' },
                  ]}
                  value={settings.streamingMode}
                  onChange={(streamingMode) =>
                    void updateSettings({ streamingMode: streamingMode as TextureStreamingMode })
                  }
                />
              ) : (
                <TextureValue value={asset.streamingMode ?? 'Not reported'} />
              ),
            },
            {
              id: 'residency',
              label: 'Residency',
              control: <TextureValue value={asset.residency ?? 'Not reported'} />,
            },
            ...(asset.hasLastGood && asset.status !== 'ready'
              ? [
                  {
                    id: 'runtime',
                    label: 'Runtime',
                    control: <TextureValue value="Using last-good cooked artifact" />,
                  },
                ]
              : []),
            {
              id: 'tile-count',
              label: 'Tile Count',
              control: (
                <TextureValue value={asset.tileCount === undefined ? 'Not reported' : String(asset.tileCount)} />
              ),
            },
            { id: 'priority', label: 'Priority', control: <TextureValue value="Not configured" /> },
            ...(asset.streamingEligibilityError
              ? [
                  {
                    id: 'eligibility',
                    label: 'Eligibility',
                    control: <TextureValue value={asset.streamingEligibilityError} />,
                  },
                ]
              : []),
          ]}
          onToggle={() => toggleSection('streaming')}
          title="Streaming"
        />

        <UiPropertyCard
          className="texture-inspector-section"
          collapsed={collapsedSections.import}
          fields={[
            { id: 'importer', label: 'Importer', control: <TextureValue value={asset.importerId ?? 'Not reported'} /> },
            { id: 'source-path', label: 'Source Path', control: <TextureValue value={asset.path} /> },
            {
              id: 'settings-version',
              label: 'Settings Version',
              control: (
                <TextureValue
                  value={
                    settings
                      ? String(settings.settingsVersion)
                      : asset.settingsVersion === undefined
                        ? 'Not reported'
                        : String(asset.settingsVersion)
                  }
                />
              ),
            },
            ...(settings
              ? [
                  {
                    id: 'max-size',
                    label: 'Max Size',
                    control: (
                      <TextureNumber
                        ariaLabel="Max Size"
                        disabled={settingsBusy}
                        max={32768}
                        min={1}
                        onChange={(maxSize) => void updateSettings({ maxSize })}
                        precision={0}
                        step={1}
                        value={settings.maxSize}
                      />
                    ),
                  },
                  {
                    id: 'power-of-two',
                    label: 'Power of Two',
                    control: (
                      <TextureSelect
                        ariaLabel="Power of Two"
                        disabled={settingsBusy}
                        options={[
                          { value: 'preserve', label: 'Preserve' },
                          { value: 'resize_down', label: 'Resize Down' },
                          { value: 'resize_up', label: 'Resize Up' },
                        ]}
                        value={settings.powerOfTwo}
                        onChange={(powerOfTwo) =>
                          void updateSettings({ powerOfTwo: powerOfTwo as TexturePowerOfTwoPolicy })
                        }
                      />
                    ),
                  },
                ]
              : []),
          ]}
          onToggle={() => toggleSection('import')}
          title="Import"
        />

        <UiPropertyCard
          className="texture-inspector-section"
          collapsed={collapsedSections.asset}
          fields={[
            { id: 'status', label: 'Status', control: <TextureValue value={asset.status} /> },
            { id: 'scope', label: 'Scope', control: <TextureValue value={asset.scope ?? 'project'} /> },
            { id: 'path', label: 'Path', control: <TextureValue value={asset.path} /> },
            ...(asset.guid ? [{ id: 'guid', label: 'GUID', control: <TextureValue value={asset.guid} /> }] : []),
          ]}
          onToggle={() => toggleSection('asset')}
          title="Asset"
        />
      </div>
    </UiPanel>
  );
}

function HorizontalRuler({ width, zoom, offset }: { width: number; zoom: number; offset: number }) {
  const marks = rulerMarks(width, zoom);
  return (
    <div
      aria-hidden="true"
      className="texture-ruler texture-ruler-horizontal"
      style={{ width: width * zoom, transform: `translateX(${offset}px)` }}
    >
      {marks.map((mark) => (
        <span
          className={mark.major ? 'texture-ruler-mark major' : 'texture-ruler-mark'}
          key={`${mark.value}-${mark.position}`}
          style={{ left: mark.position }}
        >
          {mark.major && <em>{Math.round(mark.value)}</em>}
        </span>
      ))}
    </div>
  );
}

function VerticalRuler({ height, zoom, offset }: { height: number; zoom: number; offset: number }) {
  const marks = rulerMarks(height, zoom);
  return (
    <div
      aria-hidden="true"
      className="texture-ruler texture-ruler-vertical"
      style={{ height: height * zoom, transform: `translateY(${offset}px)` }}
    >
      {marks.map((mark) => (
        <span
          className={mark.major ? 'texture-ruler-mark major' : 'texture-ruler-mark'}
          key={`${mark.value}-${mark.position}`}
          style={{ top: mark.position }}
        >
          {mark.major && <em>{Math.round(mark.value)}</em>}
        </span>
      ))}
    </div>
  );
}

export function TextureEditor({ document }: { document: EditorDocument }) {
  const asset = useMemo<AssetItem>(
    () =>
      document.assetSnapshot ?? {
        id: document.assetId ?? document.id,
        guid: document.assetGuid,
        name: document.title,
        path: document.path ?? '',
        scope: document.assetScope,
        kind: 'texture',
        status: 'unknown',
        readOnly: document.readOnly,
      },
    [document],
  );
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const panRef = useRef<PanState | null>(null);
  const resizeRef = useRef<{ startX: number; width: number } | null>(null);
  const ddsSource = extensionOf(asset.path) === 'DDS';
  const [preview, setPreview] = useState<HostAssetThumbnailSnapshot | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [analysis, setAnalysis] = useState<TexturePreviewAnalysis | null>(null);
  const [pixelReadout, setPixelReadout] = useState<string>('');
  const { settings: previewSettings } = useTextureSettings(asset.guid, asset.generation);
  const [viewport, setViewport] = useState<ViewportMetrics>({ scrollLeft: 0, scrollTop: 0, width: 0, height: 0 });
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [panning, setPanning] = useState(false);
  const [inspectorWidth, setInspectorWidth] = useState(defaultInspectorWidth);
  const viewState = useTextureEditorViewState(document.id);
  const zoom = viewState.zoom;
  const mipScale = 1 / 2 ** viewState.mipLevel;
  const displayWidth = Math.max(1, Math.round((preview?.width ?? 1) * mipScale));
  const displayHeight = Math.max(1, Math.round((preview?.height ?? 1) * mipScale));
  const renderedWidth = displayWidth * zoom;
  const renderedHeight = displayHeight * zoom;
  const canvasWidth = Math.max(viewport.width, renderedWidth + previewPadding * 2);
  const canvasHeight = Math.max(viewport.height, renderedHeight + previewPadding * 2);
  const imageLeft = Math.max(previewPadding, (canvasWidth - renderedWidth) / 2);
  const imageTop = Math.max(previewPadding, (canvasHeight - renderedHeight) / 2);

  useEffect(() => {
    let active = true;
    setPreview(null);
    setPreviewFailed(false);
    if (!asset.path || typeof window === 'undefined' || !window.arc?.host?.query) {
      setPreviewFailed(true);
      return;
    }

    void (async () => {
      try {
        const response = (await window.arc.host.query('asset.thumbnail', {
          path: asset.path,
          maxSize: 2048,
        })) as HostResponse<HostAssetThumbnailSnapshot>;
        if (!active) return;
        if (response.succeeded && response.payload?.dataUrl) setPreview(response.payload);
        else setPreviewFailed(true);
      } catch {
        if (active) setPreviewFailed(true);
      }
    })();

    return () => {
      active = false;
    };
  }, [asset.path, asset.generation]);

  useEffect(() => {
    let active = true;
    if (!preview?.dataUrl || !previewSettings) {
      setAnalysis(null);
      return;
    }
    void analyzeTexturePreview(preview.dataUrl, previewSettings)
      .then((value) => {
        if (active) setAnalysis(value);
      })
      .catch(() => {
        if (active) setAnalysis(null);
      });
    return () => {
      active = false;
    };
  }, [preview?.dataUrl, previewSettings]);

  useEffect(() => {
    const scroll = scrollRef.current;
    if (!scroll || !preview) return;

    const updateViewport = () => {
      setViewport({
        scrollLeft: scroll.scrollLeft,
        scrollTop: scroll.scrollTop,
        width: scroll.clientWidth,
        height: scroll.clientHeight,
      });
    };

    updateViewport();
    const availableWidth = Math.max(1, scroll.clientWidth - previewPadding * 2);
    const availableHeight = Math.max(1, scroll.clientHeight - previewPadding * 2);
    setTextureEditorViewState(document.id, {
      zoom: clampZoom(Math.min(1, availableWidth / displayWidth, availableHeight / displayHeight)),
    });

    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(updateViewport);
    observer.observe(scroll);
    return () => observer.disconnect();
  }, [displayHeight, displayWidth, document.id, preview]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !event.repeat) {
        setSpaceHeld(true);
        if (event.target instanceof HTMLElement && !['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
          event.preventDefault();
        }
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') setSpaceHeld(false);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', () => setSpaceHeld(false), { once: true });
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!preview) return;
    event.preventDefault();
    event.stopPropagation();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    setTextureEditorViewState(document.id, { zoom: clampZoom(zoom * factor) });
  };

  const onScroll = (event: UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    setViewport((current) => ({
      ...current,
      scrollLeft: target.scrollLeft,
      scrollTop: target.scrollTop,
      width: target.clientWidth,
      height: target.clientHeight,
    }));
  };

  const beginPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const shouldPan = event.button === 1 || (event.button === 0 && spaceHeld);
    if (!shouldPan || !scrollRef.current) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    panRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: scrollRef.current.scrollLeft,
      scrollTop: scrollRef.current.scrollTop,
    };
    setPanning(true);
  };

  const movePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    const scroll = scrollRef.current;
    if (!pan || pan.pointerId !== event.pointerId || !scroll) return;
    scroll.scrollLeft = pan.scrollLeft - (event.clientX - pan.startX);
    scroll.scrollTop = pan.scrollTop - (event.clientY - pan.startY);
  };

  const endPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (panRef.current?.pointerId !== event.pointerId) return;
    panRef.current = null;
    setPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const beginResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    resizeRef.current = { startX: event.clientX, width: inspectorWidth };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const moveResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const resize = resizeRef.current;
    if (!resize) return;
    const width = resize.width - (event.clientX - resize.startX);
    setInspectorWidth(Math.min(maxInspectorWidth, Math.max(minInspectorWidth, width)));
  };

  const inspectPixel = (event: ReactPointerEvent<HTMLImageElement>) => {
    if (!analysis) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.min(
      analysis.width - 1,
      Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * analysis.width)),
    );
    const y = Math.min(
      analysis.height - 1,
      Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * analysis.height)),
    );
    const offset = (y * analysis.width + x) * 4;
    const s = analysis.sourcePixels;
    const p = analysis.processedPixels;
    setPixelReadout(
      `${x}, ${y}  Source ${s[offset]}, ${s[offset + 1]}, ${s[offset + 2]}, ${s[offset + 3]}  Processed ${p[offset]}, ${p[offset + 1]}, ${p[offset + 2]}, ${p[offset + 3]}`,
    );
  };

  const previewDataUrl = analysis
    ? viewState.previewMode === 'source'
      ? analysis.sourceDataUrl
      : viewState.previewMode === 'difference'
        ? analysis.differenceDataUrl
        : analysis.processedDataUrl
    : preview?.dataUrl;

  const endResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    resizeRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };

  return (
    <section className="texture-editor" style={{ gridTemplateColumns: `minmax(0, 1fr) 6px ${inspectorWidth}px` }}>
      <main className="texture-preview-pane">
        <div
          className={`texture-preview-stage ${spaceHeld ? 'is-pan-ready' : ''} ${panning ? 'is-panning' : ''}`}
          onPointerCancel={endPan}
          onPointerDown={beginPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onWheel={onWheel}
        >
          <div aria-hidden="true" className="texture-ruler-corner" />
          <div aria-hidden="true" className="texture-ruler-viewport texture-ruler-horizontal-viewport">
            {preview && <HorizontalRuler width={displayWidth} zoom={zoom} offset={imageLeft - viewport.scrollLeft} />}
          </div>
          <div aria-hidden="true" className="texture-ruler-viewport texture-ruler-vertical-viewport">
            {preview && <VerticalRuler height={displayHeight} zoom={zoom} offset={imageTop - viewport.scrollTop} />}
          </div>
          <div className="texture-preview-scroll" onScroll={onScroll} ref={scrollRef}>
            <div className="texture-preview-analysis-bar" onPointerDown={(event) => event.stopPropagation()}>
              <span className="texture-preview-mode-group">
                {(['source', 'processed', 'difference'] as const).map((mode) => (
                  <UiButton
                    active={viewState.previewMode === mode}
                    key={mode}
                    onClick={() => setTextureEditorViewState(document.id, { previewMode: mode })}
                    variant="toolbar"
                  >
                    {mode[0].toUpperCase() + mode.slice(1)}
                  </UiButton>
                ))}
              </span>
              {analysis && (
                <span className="texture-preview-histogram" title="Processed RGB histogram">
                  {Array.from({ length: 32 }, (_, index) => {
                    const start = index * 8;
                    const value = Math.max(
                      ...analysis.histogram.r.slice(start, start + 8),
                      ...analysis.histogram.g.slice(start, start + 8),
                      ...analysis.histogram.b.slice(start, start + 8),
                    );
                    const peak = Math.max(1, ...analysis.histogram.r, ...analysis.histogram.g, ...analysis.histogram.b);
                    return <i key={index} style={{ height: `${Math.max(2, (value / peak) * 20)}px` }} />;
                  })}
                </span>
              )}
              <span className="texture-preview-pixel-readout">{pixelReadout || 'Hover image for pixel values'}</span>
            </div>
            {previewDataUrl && !previewFailed ? (
              <div className="texture-preview-canvas" style={{ width: canvasWidth, height: canvasHeight }}>
                <div
                  className="texture-preview-image-frame"
                  style={{
                    left: imageLeft,
                    top: imageTop,
                    width: renderedWidth,
                    height: renderedHeight,
                  }}
                >
                  <svg aria-hidden="true" className="texture-channel-filter-defs">
                    <filter id={`texture-channel-filter-${document.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`}>
                      <feColorMatrix
                        type="matrix"
                        values={`${viewState.channels.r ? 1 : 0} 0 0 0 0  0 ${viewState.channels.g ? 1 : 0} 0 0 0  0 0 ${viewState.channels.b ? 1 : 0} 0 0  0 0 0 ${viewState.channels.a ? 1 : 0} ${viewState.channels.a ? 0 : 1}`}
                      />
                    </filter>
                  </svg>
                  <img
                    alt={`${asset.name} texture preview`}
                    draggable={false}
                    height={renderedHeight}
                    onPointerMove={inspectPixel}
                    onPointerLeave={() => setPixelReadout('')}
                    src={previewDataUrl}
                    style={{ filter: `url(#texture-channel-filter-${document.id.replace(/[^a-zA-Z0-9_-]/g, '-')})` }}
                    width={renderedWidth}
                  />
                </div>
              </div>
            ) : previewFailed ? (
              <div className="texture-preview-empty">
                <Image aria-hidden="true" size={34} />
                <strong>Preview unavailable</strong>
                <span>
                  {ddsSource
                    ? 'BC-compressed DDS preview requires the block decoder; metadata and authored mip settings remain available.'
                    : 'The texture metadata is still available in the details panel.'}
                </span>
              </div>
            ) : (
              <div className="texture-preview-empty">
                <Maximize2 aria-hidden="true" size={30} />
                <strong>Loading texture…</strong>
              </div>
            )}
          </div>
        </div>
      </main>
      <div
        aria-label="Resize texture details"
        className="texture-inspector-resizer"
        onPointerCancel={endResize}
        onPointerDown={beginResize}
        onPointerMove={moveResize}
        onPointerUp={endResize}
        role="separator"
      />
      <TextureInspector asset={asset} histogram={analysis?.histogram} />
    </section>
  );
}
