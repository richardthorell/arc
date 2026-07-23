import { Mountain, Paintbrush, SlidersHorizontal, Waves } from 'lucide-react';

import { AssetThumbnail } from '../inspector/AssetPicker';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import type { HostResponse, InspectorTerrain } from '../inspector/inspectorTypes';
import type { AssetItem } from '../services/mockHost';

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

type TerrainToolsPanelProps = {
  terrain: InspectorTerrain;
  state: TerrainToolState;
  assets: ReadonlyArray<AssetItem>;
  thumbnailProvider?: AssetThumbnailProvider;
  command: (type: string, payload: unknown) => Promise<HostResponse<TerrainToolState>>;
  onStateChange: (state: TerrainToolState) => void;
  onStatus?: (message: string) => void;
};

const tools = [
  { id: 'sculpt', label: 'Raise / Lower', icon: Mountain, hint: 'Hold Shift while painting to lower terrain.' },
  { id: 'smooth', label: 'Smooth', icon: Waves, hint: 'Blend heights toward the surrounding surface.' },
  { id: 'flatten', label: 'Flatten', icon: SlidersHorizontal, hint: 'Flatten toward the height captured at stroke start.' },
] as const;

export function TerrainToolsPanel({
  terrain, state, assets, thumbnailProvider, command, onStateChange, onStatus,
}: TerrainToolsPanelProps) {
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

  const paintMode = state.tool === 'paint';
  return (
    <section className="terrain-tools-panel" aria-label="Terrain tools">
      <header className="terrain-tools-header">
        <span className="terrain-tools-mark"><Mountain size={18} /></span>
        <span><strong>Terrain Tools</strong><small>Sculpt and paint the selected terrain</small></span>
      </header>

      <div className="terrain-mode-tabs" role="tablist" aria-label="Terrain editing mode">
        <button aria-selected={!paintMode} className={!paintMode ? 'active' : ''} onClick={() => void update({ tool: 'sculpt' })}
          role="tab" type="button"><Mountain size={15} /> Sculpt</button>
        <button aria-selected={paintMode} className={paintMode ? 'active' : ''} onClick={() => void update({ tool: 'paint' })}
          role="tab" type="button"><Paintbrush size={15} /> Paint</button>
      </div>

      {!paintMode && (
        <div className="terrain-tool-grid" aria-label="Sculpt tools">
          {tools.map(({ id, label, icon: Icon, hint }) => (
            <button aria-pressed={state.tool === id} className={state.tool === id ? 'active' : ''}
              key={id} onClick={() => void update({ tool: id })} title={hint} type="button">
              <Icon size={18} /><span>{label}</span>
            </button>
          ))}
        </div>
      )}

      <div className="terrain-tool-section">
        <h3>Brush</h3>
        <TerrainRange label="Radius" max={128} min={0.25} step={0.25} suffix="m" value={state.radius}
          onChange={(radius) => void update({ radius })} />
        <TerrainRange label="Strength" max={1} min={0.001} step={0.01} value={state.strength}
          onChange={(strength) => void update({ strength })} />
        <TerrainRange label="Falloff" max={1} min={0} step={0.01} value={state.falloff}
          onChange={(falloff) => void update({ falloff })} />
        <p className="terrain-tool-hint">Use [ and ] to change radius. Alt+left or right drag orbits the view.</p>
      </div>

      {paintMode && (
        <div className="terrain-tool-section">
          <h3>Layers</h3>
          <div className="terrain-layer-grid">
            {terrain.layers.map((layer, index) => {
              const asset = assets.find((candidate) => candidate.path === layer.baseColorPath);
              return (
                <button aria-label={`Paint ${layer.name}`} aria-pressed={state.activeLayer === index}
                  className={state.activeLayer === index ? 'active' : ''} key={layer.name}
                  onClick={() => void update({ activeLayer: index, tool: 'paint' })} type="button">
                  <AssetThumbnail asset={asset} path={layer.baseColorPath} provider={thumbnailProvider} />
                  <span>{layer.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

function TerrainRange({ label, min, max, step, suffix, value, onChange }: {
  label: string;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="terrain-range">
      <span>{label}</span>
      <input aria-label={label} max={max} min={min} onChange={(event) => onChange(Number(event.target.value))}
        step={step} type="range" value={value} />
      <output>{value.toFixed(label === 'Radius' ? 1 : 2)}{suffix}</output>
    </label>
  );
}
