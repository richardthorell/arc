import { Mountain, Paintbrush, SlidersHorizontal, Waves } from 'lucide-react';
import type { PointerEvent, WheelEvent } from 'react';

import { AssetThumbnail } from '../inspector/AssetPicker';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import type { HostResponse, InspectorTerrain } from '../inspector/inspectorTypes';
import type { AssetItem } from '../services/editorHostTypes';

import './terrainEditor.css';

export type TerrainToolState = {
  entity: { index: number; generation: number };
  active: boolean;
  hoverVisible: boolean;
  tool: 'sculpt' | 'smooth' | 'flatten' | 'paint';
  radius: number;
  strength: number;
  falloff: number;
  activeLayer: number;
};

type TerrainViewportOverlayProps = {
  terrain: InspectorTerrain;
  state: TerrainToolState;
  assets: ReadonlyArray<AssetItem>;
  thumbnailProvider?: AssetThumbnailProvider;
  command: (type: string, payload: unknown) => Promise<HostResponse<TerrainToolState>>;
  onStateChange: (state: TerrainToolState) => void;
  onStatus?: (message: string) => void;
};

const sculptTools = [
  { id: 'sculpt', label: 'Raise / Lower', icon: Mountain, hint: 'Hold Shift while painting to lower terrain.' },
  { id: 'smooth', label: 'Smooth', icon: Waves, hint: 'Blend heights toward the surrounding surface.' },
  {
    id: 'flatten',
    label: 'Flatten',
    icon: SlidersHorizontal,
    hint: 'Flatten toward the height captured at stroke start.',
  },
] as const;

export function TerrainViewportOverlay({
  terrain,
  state,
  assets,
  thumbnailProvider,
  command,
  onStateChange,
  onStatus,
}: TerrainViewportOverlayProps) {
  const update = async (patch: Partial<TerrainToolState>) => {
    const next = { ...state, ...patch };
    onStateChange(next);
    const response = await command('terrain.setBrush', {
      entity: state.entity,
      tool: next.tool,
      radius: next.radius,
      strength: next.strength,
      falloff: next.falloff,
      activeLayer: next.activeLayer,
    });
    if (response.succeeded && response.payload) {
      onStateChange(response.payload);
      return;
    }
    onStateChange(state);
    onStatus?.(response.error || 'Terrain tool update failed');
  };

  const stopPointer = (event: PointerEvent<HTMLElement>) => event.stopPropagation();
  const stopWheel = (event: WheelEvent<HTMLElement>) => event.stopPropagation();
  const paintMode = state.tool === 'paint';

  return (
    <section
      className="terrain-viewport-overlay"
      aria-label="Terrain viewport tools"
      onPointerDown={stopPointer}
      onPointerMove={stopPointer}
      onPointerUp={stopPointer}
      onWheel={stopWheel}
      onContextMenu={(event) => event.stopPropagation()}
    >
      <div className="terrain-overlay-heading">
        <Mountain size={15} />
        <strong>Terrain</strong>
        <span>{paintMode ? 'Paint' : 'Sculpt'}</span>
      </div>

      <div className="terrain-mode-tabs terrain-overlay-tabs" role="tablist" aria-label="Terrain editing mode">
        <button
          aria-selected={!paintMode}
          className={!paintMode ? 'active' : ''}
          onClick={() => void update({ tool: 'sculpt' })}
          role="tab"
          type="button"
        >
          <Mountain size={14} /> Sculpt
        </button>
        <button
          aria-selected={paintMode}
          className={paintMode ? 'active' : ''}
          onClick={() => void update({ tool: 'paint' })}
          role="tab"
          type="button"
        >
          <Paintbrush size={14} /> Paint
        </button>
      </div>

      {!paintMode && (
        <div className="terrain-overlay-tool-row" aria-label="Sculpt tools">
          {sculptTools.map(({ id, label, icon: Icon, hint }) => (
            <button
              aria-label={label}
              aria-pressed={state.tool === id}
              className={state.tool === id ? 'active' : ''}
              key={id}
              onClick={() => void update({ tool: id })}
              title={hint}
              type="button"
            >
              <Icon size={15} />
              <span>{label}</span>
            </button>
          ))}
        </div>
      )}

      <div className="terrain-overlay-ranges">
        <TerrainOverlayRange
          label="Radius"
          max={128}
          min={0.25}
          step={0.25}
          suffix="m"
          value={state.radius}
          onChange={(radius) => void update({ radius })}
        />
        <TerrainOverlayRange
          label="Strength"
          max={1}
          min={0.001}
          step={0.01}
          value={state.strength}
          onChange={(strength) => void update({ strength })}
        />
        <TerrainOverlayRange
          label="Falloff"
          max={1}
          min={0}
          step={0.01}
          value={state.falloff}
          onChange={(falloff) => void update({ falloff })}
        />
      </div>

      {paintMode && (
        <div className="terrain-paint-targets" aria-label="Paint targets">
          <small>Paint target</small>
          <div className="terrain-paint-target-grid">
            {terrain.layers.map((layer, index) => {
              const asset = assets.find((candidate) => candidate.path === layer.baseColorPath);
              return (
                <button
                  aria-label={`Paint ${layer.name}`}
                  aria-pressed={state.activeLayer === index}
                  className={state.activeLayer === index ? 'active' : ''}
                  key={layer.name}
                  onClick={() => void update({ activeLayer: index, tool: 'paint' })}
                  type="button"
                >
                  <AssetThumbnail asset={asset} path={layer.baseColorPath} provider={thumbnailProvider} />
                  <span>{layer.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      <div className="terrain-overlay-hint">[ / ] radius · Shift lowers</div>
    </section>
  );
}

function TerrainOverlayRange({
  label,
  min,
  max,
  step,
  suffix,
  value,
  onChange,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="terrain-overlay-range">
      <span>{label}</span>
      <input
        aria-label={label}
        max={max}
        min={min}
        onChange={(event) => onChange(Number(event.target.value))}
        step={step}
        type="range"
        value={value}
      />
      <output>
        {Number.isInteger(value) ? value : value.toFixed(2)}
        {suffix}
      </output>
    </label>
  );
}
