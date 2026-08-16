import { useMemo, useState } from 'react';
import {
  Check,
  ChevronRight,
  MoreHorizontal,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings,
  Trash2,
} from 'lucide-react';

import {
  AssetPicker,
  MaterialPicker,
  PrefabPicker,
  TexturePicker,
} from '../inspector/AssetPicker';
import type { AssetPickerItem } from '../inspector/AssetPicker';
import { ColorControl, NumberControl, Vector3Control } from '../inspector/InspectorControls';
import { SchemaComponentCard } from '../inspector/SchemaComponents';
import type { Vec3, Vec4 } from '../inspector/inspectorTypes';
import { setPathValue } from '../inspector/propertySchema';
import type { PropertyComponentSchema } from '../inspector/propertySchema';
import { TerrainRange } from '../terrain/TerrainToolsPanel';
import {
  UiButton,
  UiIconButton,
  UiPanel,
  UiPanelHeader,
  UiSearchInput,
  UiSelectButton,
  UiTab,
  UiTabs,
  UiTextInput,
  UiTreeRow,
} from '../ui';

import '../inspector/inspector.css';
import '../tools/tools.css';
import './uiLab.css';

type DemoComponent = {
  enabled: boolean;
  name: string;
  mobility: string;
  channel: number;
  note: string;
};

const demoComponentSchema: PropertyComponentSchema<DemoComponent> = {
  id: 'ui-lab-demo',
  title: 'ExampleComponent',
  badge: 'ECS',
  fields: [
    { id: 'enabled', label: 'Enabled', path: 'enabled', type: 'boolean' },
    { id: 'name', label: 'Name', path: 'name', type: 'text' },
    {
      id: 'mobility',
      label: 'Mobility',
      path: 'mobility',
      type: 'enum',
      options: [
        { label: 'Static', value: 'static' },
        { label: 'Stationary', value: 'stationary' },
        { label: 'Movable', value: 'movable' },
      ],
    },
    {
      id: 'channel',
      label: 'Channel',
      path: 'channel',
      type: 'readonly',
      format: (value) => `Layer ${String(value ?? 0)}`,
    },
    {
      id: 'actions',
      label: 'Actions',
      path: 'note',
      type: 'actions',
      actions: [
        { id: 'rebuild', label: 'Rebuild' },
        { id: 'clear', label: 'Clear', danger: true },
      ],
    },
  ],
};

const demoAssets: AssetPickerItem[] = [
  {
    id: 'mesh-cabin',
    name: 'SM_Cabin',
    path: 'Assets/Environment/SM_Cabin.glb',
    kind: 'mesh',
    status: 'ready',
    scope: 'project',
  },
  {
    id: 'material-wood',
    name: 'M_Wood_Logs',
    path: 'Assets/Materials/M_Wood_Logs.arcmat',
    kind: 'material',
    status: 'ready',
    scope: 'project',
  },
  {
    id: 'texture-bark',
    name: 'T_Bark_Albedo',
    path: 'Assets/Textures/T_Bark_Albedo.png',
    kind: 'texture',
    status: 'ready',
    scope: 'project',
  },
  {
    id: 'prefab-cabin',
    name: 'PF_Cabin',
    path: 'Assets/Prefabs/PF_Cabin.arcprefab',
    kind: 'prefab',
    status: 'ready',
    scope: 'project',
  },
];

function LabSection({ title, description, children }: { title: string; description: string; children: React.ReactNode }) {
  return (
    <section className="ui-lab-section">
      <header className="ui-lab-section-header">
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </header>
      <div className="ui-lab-grid">{children}</div>
    </section>
  );
}

function LabCard({ title, caption, children }: { title: string; caption?: string; children: React.ReactNode }) {
  return (
    <article className="ui-lab-card">
      <header>
        <strong>{title}</strong>
        {caption && <code>{caption}</code>}
      </header>
      <div className="ui-lab-card-stage">{children}</div>
    </article>
  );
}

export function UiLab() {
  const [text, setText] = useState('Cabin_01');
  const [search, setSearch] = useState('');
  const [tab, setTab] = useState('Inspector');
  const [selectMode, setSelectMode] = useState(false);
  const [position, setPosition] = useState<Vec3>({ x: 12.5, y: 4, z: -8.25 });
  const [linkedScale, setLinkedScale] = useState(true);
  const [roughness, setRoughness] = useState(0.45);
  const [color, setColor] = useState<Vec4>({ x: 0.42, y: 0.24, z: 0.12, w: 1 });
  const [range, setRange] = useState(12);
  const [mesh, setMesh] = useState(demoAssets[0].path);
  const [material, setMaterial] = useState(demoAssets[1].path);
  const [texture, setTexture] = useState(demoAssets[2].path);
  const [prefab, setPrefab] = useState(demoAssets[3].path);
  const [componentCollapsed, setComponentCollapsed] = useState(false);
  const [component, setComponent] = useState<DemoComponent>({
    enabled: true,
    name: 'Cabin Renderable',
    mobility: 'static',
    channel: 1,
    note: '',
  });
  const [nativeEnabled, setNativeEnabled] = useState(true);
  const [nativeSelect, setNativeSelect] = useState('Default');
  const [nativeNumber, setNativeNumber] = useState(60);

  const filteredAssets = useMemo(
    () => demoAssets.filter((asset) => asset.name.toLocaleLowerCase().includes(search.toLocaleLowerCase())),
    [search],
  );

  return (
    <main className="ui-lab-shell">
      <header className="ui-lab-hero">
        <div className="ui-lab-brand">
          <span className="ui-lab-mark">A</span>
          <div>
            <strong>ARC UI Lab</strong>
            <small>Live production controls · isolated from the editor workbench</small>
          </div>
        </div>
        <div className="ui-lab-hero-actions">
          <span>Interactive</span>
          <span>Production CSS</span>
          <span>No engine host required</span>
        </div>
      </header>

      <div className="ui-lab-content">
        <LabSection title="Buttons" description="Shared button primitives and their core interaction states.">
          <LabCard title="Default" caption="UiButton">
            <div className="ui-lab-row">
              <UiButton>Default</UiButton>
              <UiButton active>Active</UiButton>
              <UiButton disabled>Disabled</UiButton>
            </div>
          </LabCard>
          <LabCard title="Primary / danger" caption="UiButton variants">
            <div className="ui-lab-row">
              <UiButton variant="primary">
                <Plus size={14} /> Create
              </UiButton>
              <UiButton variant="danger">
                <Trash2 size={14} /> Delete
              </UiButton>
            </div>
          </LabCard>
          <LabCard title="Ghost / toolbar" caption="UiButton variants">
            <div className="ui-lab-row">
              <UiButton variant="ghost">Cancel</UiButton>
              <UiButton variant="toolbar">
                <Save size={14} /> Save
              </UiButton>
            </div>
          </LabCard>
          <LabCard title="Icon button" caption="UiIconButton">
            <div className="ui-lab-row">
              <UiIconButton label="Play">
                <Play size={15} />
              </UiIconButton>
              <UiIconButton active label="Refresh">
                <RefreshCw size={15} />
              </UiIconButton>
              <UiIconButton disabled label="More actions">
                <MoreHorizontal size={15} />
              </UiIconButton>
            </div>
          </LabCard>
          <LabCard title="Select button" caption="UiSelectButton">
            <div className="ui-lab-row">
              <UiSelectButton active={selectMode} onClick={() => setSelectMode((value) => !value)}>
                Global
              </UiSelectButton>
              <UiSelectButton showChevron={false}>10°</UiSelectButton>
            </div>
          </LabCard>
        </LabSection>

        <LabSection title="Text and form inputs" description="Editor text fields plus native controls still used by production panels.">
          <LabCard title="Text input" caption="UiTextInput">
            <UiTextInput aria-label="Entity name" value={text} onChange={(event) => setText(event.target.value)} />
          </LabCard>
          <LabCard title="Search input" caption="UiSearchInput">
            <div className="ui-lab-search-wrap">
              <Search aria-hidden="true" size={14} />
              <UiSearchInput
                aria-label="Search assets"
                placeholder="Search assets…"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
              />
            </div>
          </LabCard>
          <LabCard title="Select" caption="native select">
            <select value={nativeSelect} onChange={(event) => setNativeSelect(event.target.value)}>
              <option>Default</option>
              <option>High</option>
              <option>Ultra</option>
            </select>
          </LabCard>
          <LabCard title="Checkbox" caption="native checkbox">
            <label className="ui-lab-native-check">
              <input
                checked={nativeEnabled}
                onChange={(event) => setNativeEnabled(event.target.checked)}
                type="checkbox"
              />
              <span>Enabled</span>
            </label>
          </LabCard>
          <LabCard title="Number" caption="native number">
            <input
              aria-label="Frame limit"
              min={1}
              onChange={(event) => setNativeNumber(Number(event.target.value))}
              type="number"
              value={nativeNumber}
            />
          </LabCard>
        </LabSection>

        <LabSection title="Inspector controls" description="The same controls used by schema-driven ECS component regions.">
          <LabCard title="Vector 3" caption="Vector3Control">
            <Vector3Control
              field={{ label: 'Position', precision: 2, step: 0.1, scrubSensitivity: 0.05 }}
              linked={false}
              value={position}
              onCommit={(axis, value) => setPosition((current) => ({ ...current, [axis]: value }))}
              onPreview={(axis, value) => setPosition((current) => ({ ...current, [axis]: value }))}
            />
          </LabCard>
          <LabCard title="Linked vector" caption="Vector3Control">
            <Vector3Control
              field={{ label: 'Scale', precision: 2, step: 0.1, scrubSensitivity: 0.01, linked: true }}
              linked={linkedScale}
              value={{ x: 1, y: 1, z: 1 }}
              onToggleLinked={() => setLinkedScale((value) => !value)}
              onCommit={() => undefined}
              onPreview={() => undefined}
            />
          </LabCard>
          <LabCard title="Scalar scrub" caption="NumberControl">
            <NumberControl
              field={{
                label: 'Roughness',
                precision: 2,
                step: 0.01,
                scrubSensitivity: 0.005,
                min: 0,
                max: 1,
              }}
              value={roughness}
              onCommit={setRoughness}
              onPreview={setRoughness}
            />
          </LabCard>
          <LabCard title="Color" caption="ColorControl">
            <ColorControl label="Base Color" value={color} onCommit={setColor} onPreview={setColor} />
          </LabCard>
          <LabCard title="Range + numeric" caption="TerrainRange">
            <TerrainRange label="Radius" max={128} min={0.25} step={0.25} suffix="m" value={range} onChange={setRange} />
          </LabCard>
        </LabSection>

        <LabSection title="Asset references" description="Production asset reference controls, including type-specific pickers.">
          <LabCard title="Generic asset" caption="AssetPicker">
            <AssetPicker
              assetKinds={['mesh']}
              assetTypeLabel="Mesh"
              assets={filteredAssets}
              label="Mesh"
              value={mesh}
              onChange={setMesh}
            />
          </LabCard>
          <LabCard title="Material" caption="MaterialPicker">
            <MaterialPicker assets={filteredAssets} label="Material" value={material} onChange={setMaterial} />
          </LabCard>
          <LabCard title="Texture" caption="TexturePicker">
            <TexturePicker assets={filteredAssets} label="Texture" value={texture} onChange={setTexture} />
          </LabCard>
          <LabCard title="Prefab" caption="PrefabPicker">
            <PrefabPicker assets={filteredAssets} label="Prefab" value={prefab} onChange={setPrefab} />
          </LabCard>
        </LabSection>

        <LabSection title="Navigation and containers" description="Tabs, tree rows, panels, and ECS component containers.">
          <LabCard title="Tabs" caption="UiTabs / UiTab">
            <UiTabs>
              {['Inspector', 'Lighting', 'World'].map((name) => (
                <UiTab active={tab === name} key={name} onClick={() => setTab(name)}>
                  {name}
                </UiTab>
              ))}
            </UiTabs>
          </LabCard>
          <LabCard title="Tree rows" caption="UiTreeRow">
            <div className="ui-lab-tree">
              <UiTreeRow depth={0} meta={<small>scene</small>}>
                <ChevronRight size={13} />
                <span>World</span>
              </UiTreeRow>
              <UiTreeRow depth={1} selected meta={<small>mesh</small>}>
                <Check size={13} />
                <span>Cabin_01</span>
              </UiTreeRow>
            </div>
          </LabCard>
          <LabCard title="Panel" caption="UiPanel / UiPanelHeader">
            <UiPanel className="ui-lab-demo-panel">
              <UiPanelHeader
                actions={
                  <UiIconButton label="Panel settings">
                    <Settings size={13} />
                  </UiIconButton>
                }
              >
                Example panel
              </UiPanelHeader>
              <div className="ui-lab-demo-panel-body">Panel content region</div>
            </UiPanel>
          </LabCard>
          <LabCard title="ECS component region" caption="SchemaComponentCard">
            <SchemaComponentCard
              assets={demoAssets}
              collapsed={componentCollapsed}
              context={component}
              schema={demoComponentSchema}
              onAction={() => undefined}
              onToggle={() => setComponentCollapsed((value) => !value)}
              onValue={(path, value) => setComponent((current) => setPathValue(current, path, value))}
            />
          </LabCard>
        </LabSection>
      </div>
    </main>
  );
}
