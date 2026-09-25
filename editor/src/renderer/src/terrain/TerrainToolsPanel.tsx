import { Mountain, Paintbrush, SlidersHorizontal, Waves } from 'lucide-react';

import { AssetThumbnail } from '../inspector/AssetPicker';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import type { HostResponse, InspectorTerrain } from '../inspector/inspectorTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { UiButton, UiNumericInput, UiPanelCard, UiPropertyCard, UiSlider } from '../ui';

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
  {
    id: 'flatten',
    label: 'Flatten',
    icon: SlidersHorizontal,
    hint: 'Flatten toward the height captured at stroke start.',
  },
] as const;

export function TerrainToolsPanel({
  terrain,
  state,
  assets,
  thumbnailProvider,
  command,
  onStateChange,
  onStatus,
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
        <span className="terrain-tools-mark">
          <Mountain size={18} />
        </span>
        <span>
          <strong>Terrain Tools</strong>
          <small>Sculpt and paint the selected terrain</small>
        </span>
      </header>

      <div className="terrain-mode-tabs" role="tablist" aria-label="Terrain editing mode">
        <UiButton
          active={!paintMode}
          aria-selected={!paintMode}
          onClick={() => void update({ tool: 'sculpt' })}
          role="tab"
          type="button"
          variant="toolbar"
        >
          <Mountain size={15} /> Sculpt
        </UiButton>
        <UiButton
          active={paintMode}
          aria-selected={paintMode}
          onClick={() => void update({ tool: 'paint' })}
          role="tab"
          type="button"
          variant="toolbar"
        >
          <Paintbrush size={15} /> Paint
        </UiButton>
      </div>

      {!paintMode && (
        <div className="terrain-tool-grid" aria-label="Sculpt tools">
          {tools.map(({ id, label, icon: Icon, hint }) => (
            <UiButton
              active={state.tool === id}
              aria-pressed={state.tool === id}
              key={id}
              onClick={() => void update({ tool: id })}
              title={hint}
              type="button"
            >
              <Icon size={18} />
              <span>{label}</span>
            </UiButton>
          ))}
        </div>
      )}

      <UiPropertyCard
        className="terrain-tool-section"
        expandable={false}
        fields={[
          {
            id: 'radius',
            label: 'Radius',
            control: (
              <TerrainRangeControl
                label="Radius"
                max={128}
                min={0.25}
                step={0.25}
                suffix="m"
                value={state.radius}
                onChange={(radius) => void update({ radius })}
              />
            ),
          },
          {
            id: 'strength',
            label: 'Strength',
            control: (
              <TerrainRangeControl
                label="Strength"
                max={1}
                min={0.001}
                step={0.01}
                value={state.strength}
                onChange={(strength) => void update({ strength })}
              />
            ),
          },
          {
            id: 'falloff',
            label: 'Falloff',
            control: (
              <TerrainRangeControl
                label="Falloff"
                max={1}
                min={0}
                step={0.01}
                value={state.falloff}
                onChange={(falloff) => void update({ falloff })}
              />
            ),
          },
        ]}
        title="Brush"
      />
      <p className="terrain-tool-hint">Use [ and ] to change radius. Alt + left-drag orbits the focused view.</p>

      {paintMode && (
        <UiPanelCard className="terrain-tool-section" expandable={false} title="Layers">
          <div className="terrain-layer-grid">
            {terrain.layers.map((layer, index) => {
              const asset = assets.find((candidate) => candidate.path === layer.baseColorPath);
              return (
                <UiButton
                  active={state.activeLayer === index}
                  aria-label={`Paint ${layer.name}`}
                  aria-pressed={state.activeLayer === index}
                  key={layer.name}
                  onClick={() => void update({ activeLayer: index, tool: 'paint' })}
                  type="button"
                  variant="ghost"
                >
                  <AssetThumbnail asset={asset} path={layer.baseColorPath} provider={thumbnailProvider} />
                  <span>{layer.name}</span>
                </UiButton>
              );
            })}
          </div>
        </UiPanelCard>
      )}
    </section>
  );
}

export function TerrainRangeControl({
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
  const precision = Math.max(0, `${step}`.split('.')[1]?.length ?? 0);
  return (
    <span className="terrain-range-control">
      <UiSlider aria-label={label} max={max} min={min} step={step} value={value} onValueChange={onChange} />
      <UiNumericInput
        ariaLabel={`${label} numeric value`}
        max={max}
        min={min}
        onCommit={onChange}
        precision={precision}
        scrubSensitivity={step}
        step={step}
        unit={suffix}
        value={value}
      />
    </span>
  );
}

export const TerrainRange = TerrainRangeControl;
