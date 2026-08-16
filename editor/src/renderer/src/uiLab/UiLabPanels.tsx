import { useMemo, useState } from 'react';
import {
  Box,
  ChevronDown,
  ChevronRight,
  Circle,
  Eye,
  File,
  Folder,
  FolderOpen,
  MoreHorizontal,
  Plus,
  Search,
  Settings,
  Sun,
} from 'lucide-react';

import { panelRegistry } from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';
import {
  UiButton,
  UiIconButton,
  UiPanel,
  UiPanelHeader,
  UiSearchInput,
  UiSelect,
  UiTextInput,
  UiTreeRow,
} from '../ui';

import './uiLabPanels.css';

const panelOrder: WorkbenchPanelId[] = [
  'viewport',
  'hierarchy',
  'inspector',
  'assetExplorer',
  'search',
  'renderGraph',
  'shaderEditor',
  'lighting',
  'worldSettings',
  'contentBrowser',
  'console',
  'buildOutput',
  'versionControl',
  'aiAssistant',
  'profiler',
  'settings',
];

const panelSize: Partial<Record<WorkbenchPanelId, 'featured' | 'tall' | 'normal'>> = {
  viewport: 'featured',
  hierarchy: 'tall',
  inspector: 'tall',
  contentBrowser: 'featured',
  renderGraph: 'featured',
};

function PanelFrame({ id, children }: { id: WorkbenchPanelId; children: React.ReactNode }) {
  const panel = panelRegistry[id];
  const Icon = panel.icon;
  const size = panelSize[id] ?? 'normal';

  return (
    <article className={`ui-lab-panel-card ui-lab-panel-card-${size}`} data-panel-id={id}>
      <UiPanel className="ui-lab-panel-preview">
        <UiPanelHeader
          className="ui-lab-panel-preview-header"
          actions={
            <UiIconButton label={`${panel.title} options`} variant="ghost">
              <MoreHorizontal size={14} />
            </UiIconButton>
          }
        >
          <span className="ui-lab-panel-title">
            <Icon size={14} aria-hidden="true" />
            {panel.title}
          </span>
        </UiPanelHeader>
        <div className="ui-lab-panel-body">{children}</div>
      </UiPanel>
      <footer>
        <code>{id}</code>
        <span>{panel.defaultRegion}</span>
      </footer>
    </article>
  );
}

function ViewportPreview() {
  const [view, setView] = useState('Perspective');
  return (
    <div className="ui-lab-viewport-preview">
      <div className="ui-lab-viewport-toolbar">
        <UiSelect
          ariaLabel="Viewport projection"
          options={[
            { label: 'Perspective', value: 'Perspective' },
            { label: 'Top', value: 'Top' },
            { label: 'Front', value: 'Front' },
          ]}
          value={view}
          onValueChange={setView}
        />
        <UiButton variant="toolbar">Lit</UiButton>
        <UiIconButton label="Viewport options" variant="toolbar">
          <Settings size={14} />
        </UiIconButton>
      </div>
      <div className="ui-lab-viewport-gizmo" aria-hidden="true">
        <span className="axis-x">X</span>
        <span className="axis-y">Y</span>
        <span className="axis-z">Z</span>
      </div>
      <div className="ui-lab-viewport-status">Perspective · Lit · 60 FPS</div>
    </div>
  );
}

function HierarchyPreview() {
  const [selected, setSelected] = useState('cabin');
  return (
    <div className="ui-lab-hierarchy-preview">
      <div className="ui-lab-panel-toolbar">
        <div className="ui-lab-search-wrap">
          <Search size={14} aria-hidden="true" />
          <UiSearchInput aria-label="Search hierarchy" placeholder="Search hierarchy…" />
        </div>
        <UiIconButton label="Create entity" variant="toolbar">
          <Plus size={14} />
        </UiIconButton>
      </div>
      <div className="ui-lab-tree">
        <UiTreeRow as="div">
          <ChevronDown size={13} />
          <FolderOpen size={14} />
          <span>MountainVillage</span>
          <small>SCENE</small>
        </UiTreeRow>
        <UiTreeRow depth={1} selected={selected === 'cabin'} onClick={() => setSelected('cabin')}>
          <ChevronRight size={13} />
          <Box size={14} />
          <span>Cabin_01</span>
          <Eye size={13} />
        </UiTreeRow>
        <UiTreeRow depth={1} selected={selected === 'sun'} onClick={() => setSelected('sun')}>
          <span />
          <Sun size={14} />
          <span>Sun</span>
          <Eye size={13} />
        </UiTreeRow>
        <UiTreeRow depth={1} selected={selected === 'camera'} onClick={() => setSelected('camera')}>
          <span />
          <Box size={14} />
          <span>Main Camera</span>
          <Eye size={13} />
        </UiTreeRow>
        <UiTreeRow depth={1} as="div">
          <ChevronDown size={13} />
          <Folder size={14} />
          <span>Environment</span>
          <small>6</small>
        </UiTreeRow>
        <UiTreeRow depth={2} as="div">
          <span />
          <Box size={14} />
          <span>PineCluster_A</span>
        </UiTreeRow>
        <UiTreeRow depth={2} as="div">
          <span />
          <Box size={14} />
          <span>RockFormation_02</span>
        </UiTreeRow>
      </div>
    </div>
  );
}

function InspectorPreview() {
  const [mobility, setMobility] = useState('Static');
  return (
    <div className="ui-lab-inspector-preview">
      <div className="ui-lab-inspector-entity">
        <div className="ui-lab-inspector-icon"><Box size={18} /></div>
        <div>
          <strong>Cabin_01</strong>
          <small>Entity 1842:7</small>
        </div>
      </div>
      <section className="ui-lab-inspector-section">
        <header><ChevronDown size={13} /> Transform</header>
        <div className="ui-lab-inspector-row"><span>Position</span><div className="ui-lab-vector"><b>X</b><UiTextInput value="12.50" readOnly /><b>Y</b><UiTextInput value="4.00" readOnly /><b>Z</b><UiTextInput value="-8.25" readOnly /></div></div>
        <div className="ui-lab-inspector-row"><span>Rotation</span><div className="ui-lab-vector"><b>X</b><UiTextInput value="0.0" readOnly /><b>Y</b><UiTextInput value="35.0" readOnly /><b>Z</b><UiTextInput value="0.0" readOnly /></div></div>
        <div className="ui-lab-inspector-row"><span>Scale</span><div className="ui-lab-vector"><b>X</b><UiTextInput value="1.00" readOnly /><b>Y</b><UiTextInput value="1.00" readOnly /><b>Z</b><UiTextInput value="1.00" readOnly /></div></div>
      </section>
      <section className="ui-lab-inspector-section">
        <header><ChevronDown size={13} /> Renderable</header>
        <div className="ui-lab-inspector-row"><span>Mesh</span><UiTextInput value="SM_Cabin" readOnly /></div>
        <div className="ui-lab-inspector-row"><span>Material</span><UiTextInput value="M_Wood_Logs" readOnly /></div>
        <div className="ui-lab-inspector-row"><span>Mobility</span><UiSelect ariaLabel="Mobility" options={[{ label: 'Static', value: 'Static' }, { label: 'Stationary', value: 'Stationary' }, { label: 'Movable', value: 'Movable' }]} value={mobility} onValueChange={setMobility} /></div>
      </section>
    </div>
  );
}

function AssetExplorerPreview() {
  return (
    <div className="ui-lab-asset-explorer-preview">
      <div className="ui-lab-panel-toolbar"><UiButton variant="toolbar">Project</UiButton><UiButton variant="ghost">Built-in</UiButton></div>
      <UiTreeRow as="div"><ChevronDown size={13} /><FolderOpen size={14} /><span>Assets</span></UiTreeRow>
      <UiTreeRow depth={1} as="div"><ChevronDown size={13} /><FolderOpen size={14} /><span>Environment</span></UiTreeRow>
      <UiTreeRow depth={2} as="div"><span /><Folder size={14} /><span>Cabins</span><small>8</small></UiTreeRow>
      <UiTreeRow depth={2} as="div"><span /><Folder size={14} /><span>Foliage</span><small>23</small></UiTreeRow>
      <UiTreeRow depth={1} as="div"><span /><Folder size={14} /><span>Materials</span><small>14</small></UiTreeRow>
      <UiTreeRow depth={1} as="div"><span /><Folder size={14} /><span>Textures</span><small>31</small></UiTreeRow>
    </div>
  );
}

function SearchPreview() {
  return (
    <div className="ui-lab-search-panel-preview">
      <div className="ui-lab-search-wrap"><Search size={14} /><UiSearchInput aria-label="Global search" value="cabin" readOnly /></div>
      <div className="ui-lab-result-group"><strong>Entities</strong><button><Box size={14} />Cabin_01 <small>MountainVillage</small></button></div>
      <div className="ui-lab-result-group"><strong>Assets</strong><button><File size={14} />SM_Cabin.glb <small>Environment/Cabins</small></button><button><File size={14} />M_CabinRoof.arcmat <small>Materials</small></button></div>
    </div>
  );
}

function RenderGraphPreview() {
  return (
    <div className="ui-lab-render-graph-preview">
      <div className="ui-lab-graph-grid" />
      <div className="ui-lab-graph-node node-a"><strong>Depth Prepass</strong><span>Depth</span><i /></div>
      <div className="ui-lab-graph-node node-b"><strong>GBuffer</strong><span>Albedo · Normal · ORM</span><i /></div>
      <div className="ui-lab-graph-node node-c"><strong>Lighting</strong><span>HDR Color</span><i /></div>
      <div className="ui-lab-graph-edge edge-a" />
      <div className="ui-lab-graph-edge edge-b" />
    </div>
  );
}

function ShaderEditorPreview() {
  return (
    <div className="ui-lab-code-preview">
      <div className="ui-lab-code-gutter">1<br />2<br />3<br />4<br />5<br />6<br />7</div>
      <pre><span className="kw">float4</span> PSMain(VSOutput input) : SV_Target {'{'}{`\n`}  <span className="kw">float3</span> N = normalize(input.normal);{`\n`}  <span className="kw">float3</span> L = normalize(sunDirection);{`\n`}  <span className="kw">float</span> NoL = saturate(dot(N, L));{`\n`}  <span className="cm">// ARC physically based lighting</span>{`\n`}  <span className="kw">return</span> float4(baseColor * NoL, 1.0);{`\n`}{'}'}</pre>
    </div>
  );
}

function LightingPreview() {
  return (
    <div className="ui-lab-form-panel">
      <label><span>Environment</span><UiSelect ariaLabel="Environment" value="HDRI" options={[{ label: 'HDRI', value: 'HDRI' }, { label: 'Physical Sky', value: 'Physical Sky' }]} onValueChange={() => undefined} /></label>
      <label><span>Intensity</span><UiTextInput value="1.25" readOnly /></label>
      <label><span>Exposure</span><input type="range" min="0" max="100" defaultValue="58" /></label>
      <label><span>Sun</span><UiButton variant="toolbar"><Sun size={14} /> Directional Light</UiButton></label>
    </div>
  );
}

function WorldSettingsPreview() {
  return (
    <div className="ui-lab-form-panel">
      <label><span>Gravity</span><UiTextInput value="0, -9.81, 0" readOnly /></label>
      <label><span>Time Scale</span><UiTextInput value="1.0" readOnly /></label>
      <label><span>Default Layer</span><UiSelect ariaLabel="Default layer" value="World" options={[{ label: 'World', value: 'World' }, { label: 'Gameplay', value: 'Gameplay' }]} onValueChange={() => undefined} /></label>
      <label className="ui-lab-check-row"><span>Streaming</span><input type="checkbox" defaultChecked /></label>
    </div>
  );
}

const assetTiles = [
  ['SM_Cabin', 'GLB'],
  ['M_Wood_Logs', 'MAT'],
  ['T_Bark_Albedo', 'PNG'],
  ['PF_Cabin', 'PREFAB'],
  ['SM_Pine_A', 'GLB'],
  ['M_Rock', 'MAT'],
];

function ContentBrowserPreview() {
  const [filter, setFilter] = useState('');
  const visible = useMemo(() => assetTiles.filter(([name]) => name.toLowerCase().includes(filter.toLowerCase())), [filter]);
  return (
    <div className="ui-lab-content-browser-preview">
      <div className="ui-lab-panel-toolbar"><div className="ui-lab-breadcrumb">Assets <ChevronRight size={13} /> Environment</div><div className="ui-lab-search-wrap"><Search size={14} /><UiSearchInput aria-label="Filter content" placeholder="Filter…" value={filter} onChange={(event) => setFilter(event.target.value)} /></div></div>
      <div className="ui-lab-asset-grid">{visible.map(([name, kind]) => <button key={name} className="ui-lab-asset-tile"><span className="ui-lab-asset-thumb"><Box size={24} /></span><strong>{name}</strong><small>{kind}</small></button>)}</div>
    </div>
  );
}

function ConsolePreview() {
  return <div className="ui-lab-console-preview"><div><time>23:17:04</time><span className="info">INFO</span><p>Project MountainVillage opened.</p></div><div><time>23:17:05</time><span className="info">INFO</span><p>Imported SM_Cabin.glb in 42 ms.</p></div><div><time>23:17:06</time><span className="warn">WARN</span><p>Shadow atlas occupancy reached 78%.</p></div><div><time>23:17:08</time><span className="info">INFO</span><p>Frame graph compiled: 18 passes.</p></div></div>;
}

function BuildOutputPreview() {
  return <div className="ui-lab-build-preview"><div className="ui-lab-build-summary"><strong>Development · Windows</strong><span>Success · 4.8s</span></div><ol><li className="done"><Circle /> Configure project <small>0.4s</small></li><li className="done"><Circle /> Compile game module <small>2.6s</small></li><li className="done"><Circle /> Package assets <small>1.2s</small></li><li className="done"><Circle /> Write manifest <small>0.6s</small></li></ol></div>;
}

function VersionControlPreview() {
  return <div className="ui-lab-vcs-preview"><div className="ui-lab-vcs-branch">main <small>3 changes</small></div><button><span className="modified">M</span><p>Cabin.scene<small>Scenes/MountainVillage</small></p></button><button><span className="modified">M</span><p>M_Wood_Logs.arcmat<small>Assets/Materials</small></p></button><button><span className="added">A</span><p>PF_Cabin.arcprefab<small>Assets/Prefabs</small></p></button></div>;
}

function AiPreview() {
  return <div className="ui-lab-ai-preview"><div className="ui-lab-ai-status"><span /> Gateway connected <small>localhost:7777</small></div><div className="ui-lab-ai-message user">Why is Cabin_01 not casting a shadow?</div><div className="ui-lab-ai-message assistant">The mesh is using a material with shadow casting disabled. I can inspect or update the component.</div><div className="ui-lab-ai-compose"><UiTextInput placeholder="Ask ARC…" /><UiButton variant="primary">Send</UiButton></div></div>;
}

function ProfilerPreview() {
  const bars = [42, 68, 51, 74, 62, 88, 57, 66, 49, 72, 61, 79, 55, 64, 58, 71];
  return <div className="ui-lab-profiler-preview"><div className="ui-lab-profiler-metrics"><span><strong>6.8 ms</strong>CPU</span><span><strong>8.4 ms</strong>GPU</span><span><strong>119 FPS</strong>Frame</span></div><div className="ui-lab-profiler-chart">{bars.map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}</div><div className="ui-lab-profiler-legend"><span>Renderer 3.8 ms</span><span>Physics 1.1 ms</span><span>Game 1.4 ms</span></div></div>;
}

function SettingsPreview() {
  return <div className="ui-lab-settings-preview"><aside><button className="active">Editor</button><button>Viewport</button><button>Assets</button><button>Shortcuts</button></aside><div className="ui-lab-form-panel"><label><span>Auto Save</span><input type="checkbox" defaultChecked /></label><label><span>Interval</span><UiSelect ariaLabel="Autosave interval" value="5 minutes" options={[{ label: '5 minutes', value: '5 minutes' }, { label: '10 minutes', value: '10 minutes' }]} onValueChange={() => undefined} /></label><label><span>Recent Projects</span><UiTextInput value="10" readOnly /></label></div></div>;
}

function PanelBody({ id }: { id: WorkbenchPanelId }) {
  switch (id) {
    case 'viewport': return <ViewportPreview />;
    case 'hierarchy': return <HierarchyPreview />;
    case 'inspector': return <InspectorPreview />;
    case 'assetExplorer': return <AssetExplorerPreview />;
    case 'search': return <SearchPreview />;
    case 'renderGraph': return <RenderGraphPreview />;
    case 'shaderEditor': return <ShaderEditorPreview />;
    case 'lighting': return <LightingPreview />;
    case 'worldSettings': return <WorldSettingsPreview />;
    case 'contentBrowser': return <ContentBrowserPreview />;
    case 'console': return <ConsolePreview />;
    case 'buildOutput': return <BuildOutputPreview />;
    case 'versionControl': return <VersionControlPreview />;
    case 'aiAssistant': return <AiPreview />;
    case 'profiler': return <ProfilerPreview />;
    case 'settings': return <SettingsPreview />;
  }
}

export function UiLabPanels() {
  return (
    <main className="ui-lab-panels-shell">
      <header className="ui-lab-panels-hero">
        <div>
          <strong>Panel Lab</strong>
          <span>Production panel shells with host-independent fixture content.</span>
        </div>
        <small>{panelOrder.length} registered panels</small>
      </header>
      <section className="ui-lab-panels-grid" aria-label="Editor panel gallery">
        {panelOrder.map((id) => (
          <PanelFrame id={id} key={id}>
            <PanelBody id={id} />
          </PanelFrame>
        ))}
      </section>
    </main>
  );
}
