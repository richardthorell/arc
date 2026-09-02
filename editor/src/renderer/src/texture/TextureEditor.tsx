import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, UIEvent, WheelEvent } from 'react';
import { Image, Maximize2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { UiButton, UiPanel, UiPanelSection } from '../ui';
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

function TextureProperty({ label, value }: { label: string; value: string }) {
  return (
    <div className="inspector-property texture-inspector-property">
      <span className="inspector-property-label">{label}</span>
      <span className="texture-inspector-value" title={value}>
        {value}
      </span>
    </div>
  );
}

function TextureInspector({ asset, histogram }: { asset: AssetItem; histogram?: TexturePreviewAnalysis['histogram'] }) {
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

  return (
    <UiPanel aria-label="Texture details" className="texture-inspector" role="complementary" variant="inspector">
      <div className="texture-inspector-sections">
        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.texture}
          onToggle={() => toggleSection('texture')}
          title="Texture"
        >
          <TextureProperty label="Type" value={textureTypeOf(asset)} />
          <TextureProperty label="Dimensions" value={dimensionsOf(asset)} />
          <TextureProperty label="Depth / Layers" value={asset.depth === undefined ? '1' : String(asset.depth)} />
          <TextureProperty label="Format" value={asset.textureFormat ?? extensionOf(asset.path)} />
          {settings ? (
            <>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Preset</span>
                <select
                  aria-label="Texture preset"
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  onChange={(event) => void updateSettings({ preset: event.target.value as TexturePreset })}
                  value={settings.preset}
                >
                  <option value="custom">Custom</option>
                  <option value="color">Color</option>
                  <option value="normal_map">Normal Map</option>
                  <option value="data">Data / Mask</option>
                  <option value="hdr">HDR</option>
                  <option value="ui">UI</option>
                  <option value="environment">Environment</option>
                </select>
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Semantic</span>
                <select
                  aria-label="Texture semantic"
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  onChange={(event) => void updateSettings({ semantic: event.target.value as TextureSemantic })}
                  value={settings.semantic}
                >
                  <option value="generic_color">Generic Color</option>
                  <option value="base_color">Base Color</option>
                  <option value="emissive">Emissive</option>
                  <option value="normal">Normal</option>
                  <option value="metallic_roughness">Metallic / Roughness</option>
                  <option value="occlusion">Occlusion</option>
                  <option value="clear_coat">Clear Coat</option>
                  <option value="anisotropy">Anisotropy</option>
                  <option value="thickness">Thickness</option>
                  <option value="transmission">Transmission</option>
                  <option value="lightmap">Lightmap</option>
                  <option value="environment">Environment</option>
                </select>
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Color Space</span>
                <select
                  aria-label="Texture color space"
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  onChange={(event) => void updateSettings({ colorSpace: event.target.value as TextureColorSpace })}
                  value={settings.colorSpace}
                >
                  <option value="srgb">sRGB</option>
                  <option value="linear">Linear</option>
                </select>
              </label>
            </>
          ) : (
            <TextureProperty label="Color Space" value={settingsError ?? 'Loading…'} />
          )}
          <TextureProperty label="Alpha" value="Not reported" />
          <TextureProperty label="Source Size" value={formatBytes(asset.sourceBytes)} />
        </UiPanelSection>

        <TextureStage3Controls asset={asset} />
        <TextureCurveControls asset={asset} histogram={histogram} />

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.sampling}
          onToggle={() => toggleSection('sampling')}
          title="Sampling"
        >
          {settings ? (
            <>
              {(['wrapU', 'wrapV'] as const).map((field) => (
                <label className="inspector-property texture-inspector-property" key={field}>
                  <span className="inspector-property-label">{field === 'wrapU' ? 'Wrap U' : 'Wrap V'}</span>
                  <select
                    className="texture-inspector-select"
                    disabled={settingsBusy}
                    value={settings[field]}
                    onChange={(event) => void updateSettings({ [field]: event.target.value as TextureAddressMode })}
                  >
                    <option value="repeat">Repeat</option>
                    <option value="clamp_to_edge">Clamp</option>
                    <option value="mirrored_repeat">Mirror</option>
                  </select>
                </label>
              ))}
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Min Filter</span>
                <select
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  value={settings.minFilter}
                  onChange={(event) => void updateSettings({ minFilter: event.target.value as TextureFilterMode })}
                >
                  <option value="linear">Linear</option>
                  <option value="nearest">Nearest</option>
                </select>
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Mag Filter</span>
                <select
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  value={settings.magFilter}
                  onChange={(event) => void updateSettings({ magFilter: event.target.value as TextureFilterMode })}
                >
                  <option value="linear">Linear</option>
                  <option value="nearest">Nearest</option>
                </select>
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Mip Filter</span>
                <select
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  value={settings.mipFilter}
                  onChange={(event) => void updateSettings({ mipFilter: event.target.value as TextureMipFilterMode })}
                >
                  <option value="linear">Linear</option>
                  <option value="nearest">Nearest</option>
                </select>
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Anisotropy</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy}
                  max={16}
                  min={1}
                  onChange={(event) => void updateSettings({ anisotropy: Number(event.target.value) })}
                  step={1}
                  type="number"
                  value={settings.anisotropy}
                />
              </label>
            </>
          ) : (
            <TextureProperty label="Sampling" value={settingsError ?? 'Loading…'} />
          )}
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.mipmaps}
          onToggle={() => toggleSection('mipmaps')}
          title="Mipmaps"
        >
          {settings && (
            <>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Generate Mips</span>
                <input
                  checked={settings.generateMips}
                  disabled={settingsBusy}
                  onChange={(event) => void updateSettings({ generateMips: event.target.checked })}
                  type="checkbox"
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Generation Filter</span>
                <select
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  value={settings.mipGenerationFilter}
                  onChange={(event) =>
                    void updateSettings({ mipGenerationFilter: event.target.value as TextureMipGenerationFilter })
                  }
                >
                  <option value="kaiser">Kaiser</option>
                  <option value="lanczos">Lanczos</option>
                  <option value="bicubic">Bicubic</option>
                  <option value="bilinear">Bilinear</option>
                  <option value="box">Box</option>
                  <option value="nearest">Nearest</option>
                </select>
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Sharpen</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy || !settings.generateMips}
                  max={2}
                  min={0}
                  step={0.05}
                  type="number"
                  value={settings.mipSharpen}
                  onChange={(event) => void updateSettings({ mipSharpen: Number(event.target.value) })}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Dither</span>
                <input
                  checked={settings.ditherMips}
                  disabled={settingsBusy || !settings.generateMips}
                  type="checkbox"
                  onChange={(event) => void updateSettings({ ditherMips: event.target.checked })}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">De-band</span>
                <input
                  checked={settings.debandMips}
                  disabled={settingsBusy || !settings.generateMips}
                  type="checkbox"
                  onChange={(event) => void updateSettings({ debandMips: event.target.checked })}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">De-band Strength</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy || !settings.generateMips || !settings.debandMips}
                  max={1}
                  min={0}
                  step={0.05}
                  type="number"
                  value={settings.debandStrength}
                  onChange={(event) => void updateSettings({ debandStrength: Number(event.target.value) })}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Preserve Alpha</span>
                <input
                  checked={settings.preserveAlphaCoverage}
                  disabled={settingsBusy}
                  onChange={(event) => void updateSettings({ preserveAlphaCoverage: event.target.checked })}
                  type="checkbox"
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Alpha Threshold</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy || !settings.preserveAlphaCoverage}
                  max={1}
                  min={0}
                  onChange={(event) => void updateSettings({ alphaCoverageThreshold: Number(event.target.value) })}
                  step={0.05}
                  type="number"
                  value={settings.alphaCoverageThreshold}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">LOD Bias</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy}
                  onChange={(event) => void updateSettings({ lodBias: Number(event.target.value) })}
                  step={0.25}
                  type="number"
                  value={settings.lodBias}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Min LOD</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy}
                  min={0}
                  onChange={(event) => void updateSettings({ minimumLod: Number(event.target.value) })}
                  step={0.25}
                  type="number"
                  value={settings.minimumLod}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Max LOD</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy}
                  min={0}
                  onChange={(event) => void updateSettings({ maximumLod: Number(event.target.value) })}
                  step={0.25}
                  type="number"
                  value={settings.maximumLod}
                />
              </label>
            </>
          )}
          <TextureProperty
            label="Mip Count"
            value={asset.mipLevels === undefined ? 'Not reported' : String(asset.mipLevels)}
          />
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.compression}
          onToggle={() => toggleSection('compression')}
          title="Compression"
        >
          {settings && (
            <label className="inspector-property texture-inspector-property">
              <span className="inspector-property-label">Policy</span>
              <select
                className="texture-inspector-select"
                disabled={settingsBusy}
                value={settings.compression}
                onChange={(event) =>
                  void updateSettings({ compression: event.target.value as TextureCompressionPolicy })
                }
              >
                <option value="automatic">Automatic</option>
                <option value="color">Color</option>
                <option value="normal">Normal</option>
                <option value="mask">Mask</option>
                <option value="hdr">HDR</option>
                <option value="uncompressed">Uncompressed</option>
              </select>
            </label>
          )}
          <TextureProperty label="GPU Format" value={asset.textureFormat ?? 'Resolved at cook'} />
          <TextureProperty label="Artifact Size" value={formatBytes(asset.artifactSize)} />
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.streaming}
          onToggle={() => toggleSection('streaming')}
          title="Streaming"
        >
          {settings ? (
            <label className="inspector-property texture-inspector-property">
              <span className="inspector-property-label">Mode</span>
              <select
                aria-label="Texture streaming mode"
                className="texture-inspector-select"
                disabled={settingsBusy}
                onChange={(event) => void updateSettings({ streamingMode: event.target.value as TextureStreamingMode })}
                value={settings.streamingMode}
              >
                <option value="resident">Resident</option>
                <option value="streamed_mips">Streamed Mips</option>
                <option value="virtual_tiles">Virtual Tiles</option>
              </select>
            </label>
          ) : (
            <TextureProperty label="Mode" value={asset.streamingMode ?? 'Not reported'} />
          )}
          <TextureProperty label="Residency" value={asset.residency ?? 'Not reported'} />
          <TextureProperty
            label="Tile Count"
            value={asset.tileCount === undefined ? 'Not reported' : String(asset.tileCount)}
          />
          <TextureProperty label="Priority" value="Not configured" />
          {asset.streamingEligibilityError && (
            <TextureProperty label="Eligibility" value={asset.streamingEligibilityError} />
          )}
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.import}
          onToggle={() => toggleSection('import')}
          title="Import"
        >
          <TextureProperty label="Importer" value={asset.importerId ?? 'Not reported'} />
          <TextureProperty label="Source Path" value={asset.path} />
          <TextureProperty
            label="Settings Version"
            value={
              settings
                ? String(settings.settingsVersion)
                : asset.settingsVersion === undefined
                  ? 'Not reported'
                  : String(asset.settingsVersion)
            }
          />
          {settings && (
            <>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Max Size</span>
                <input
                  className="texture-inspector-input"
                  disabled={settingsBusy}
                  max={32768}
                  min={1}
                  onChange={(event) => void updateSettings({ maxSize: Number(event.target.value) })}
                  step={1}
                  type="number"
                  value={settings.maxSize}
                />
              </label>
              <label className="inspector-property texture-inspector-property">
                <span className="inspector-property-label">Power of Two</span>
                <select
                  className="texture-inspector-select"
                  disabled={settingsBusy}
                  value={settings.powerOfTwo}
                  onChange={(event) =>
                    void updateSettings({ powerOfTwo: event.target.value as TexturePowerOfTwoPolicy })
                  }
                >
                  <option value="preserve">Preserve</option>
                  <option value="resize_down">Resize Down</option>
                  <option value="resize_up">Resize Up</option>
                </select>
              </label>
            </>
          )}
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.asset}
          onToggle={() => toggleSection('asset')}
          title="Asset"
        >
          <TextureProperty label="Status" value={asset.status} />
          <TextureProperty label="Scope" value={asset.scope ?? 'project'} />
          <TextureProperty label="Path" value={asset.path} />
          {asset.guid && <TextureProperty label="GUID" value={asset.guid} />}
        </UiPanelSection>
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
                <span>The texture metadata is still available in the details panel.</span>
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
