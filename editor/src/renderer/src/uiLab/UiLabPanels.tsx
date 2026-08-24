import { useMemo, useState } from 'react';

import { ExplorerPanel } from '../app/Workbench';
import { panelRegistry } from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';
import { AiGatewayPanel } from '../ai/AiGatewayPanel';
import { BuildOutputPanel } from '../buildOutput/BuildOutputPanel';
import { ConsolePanel } from '../console/ConsolePanel';
import { ContentBrowserPanel } from '../content/ContentBrowserPanel';
import { WorldEnvironmentInspector } from '../environment/WorldEnvironmentInspector';
import { InspectorPanel } from '../inspector/InspectorPanel';
import { LightingPanel } from '../lighting/LightingPanel';
import { ProfilerPanel } from '../profiler/ProfilerPanel';
import { RenderGraphPanel } from '../renderGraph/RenderGraphPanel';
import { SearchPanel } from '../search/SearchPanel';
import { ShaderEditorPanel } from '../shader/ShaderEditorPanel';
import { VersionControlPanel } from '../versionControl/VersionControlPanel';
import { ViewportPanel } from '../viewport/ViewportPanel';

import {
  panelBuildFixture,
  panelDiagnosticsFixture,
  panelGatewayFixture,
  panelInspectorFixture,
  panelProfilerFixtures,
  panelProjectFixture,
  panelWorldEnvironmentFixture,
} from './UiLabPanelFixtures';

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
];

const panelSize: Partial<Record<WorkbenchPanelId, 'featured' | 'tall' | 'normal'>> = {
  viewport: 'featured',
  hierarchy: 'tall',
  inspector: 'tall',
  renderGraph: 'featured',
  worldSettings: 'tall',
  contentBrowser: 'featured',
  profiler: 'featured',
};

const productionComponentNames: Partial<Record<WorkbenchPanelId, string>> = {
  viewport: 'ViewportPanel',
  hierarchy: 'ExplorerPanel',
  inspector: 'InspectorPanel',
  search: 'SearchPanel',
  renderGraph: 'RenderGraphPanel',
  shaderEditor: 'ShaderEditorPanel',
  lighting: 'LightingPanel',
  worldSettings: 'WorldEnvironmentInspector',
  contentBrowser: 'ContentBrowserPanel',
  console: 'ConsolePanel',
  buildOutput: 'BuildOutputPanel',
  versionControl: 'VersionControlPanel',
  aiAssistant: 'AiGatewayPanel',
  profiler: 'ProfilerPanel',
};

function PanelCard({ id, children }: { id: WorkbenchPanelId; children: React.ReactNode }) {
  const descriptor = panelRegistry[id];
  const size = panelSize[id] ?? 'normal';
  const componentName = productionComponentNames[id];

  return (
    <article className={`ui-lab-production-panel ui-lab-production-panel-${size}`} data-panel-id={id}>
      <header className="ui-lab-production-panel-label">
        <span>
          <strong>{descriptor.title}</strong>
          <small>{descriptor.defaultRegion} region</small>
        </span>
        <code>{componentName ?? 'Workbench internal'}</code>
      </header>
      <div className="ui-lab-production-panel-stage">{children}</div>
    </article>
  );
}

function InternalPanelNotice({ id }: { id: WorkbenchPanelId }) {
  return (
    <div className="ui-lab-internal-panel-notice">
      <strong>{panelRegistry[id].title} is currently private to Workbench.tsx</strong>
      <span>
        The UI Lab intentionally does not duplicate its markup. Extract it into a production component before styling it
        here.
      </span>
    </div>
  );
}

export function UiLabPanels() {
  const [selectedEntityId, setSelectedEntityId] = useState('1842:7');
  const [selectedEntityIds, setSelectedEntityIds] = useState<ReadonlySet<string>>(() => new Set(['1842:7']));
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>('mesh-cabin');
  const [consoleLocked, setConsoleLocked] = useState(true);
  const [clearedConsoleIds, setClearedConsoleIds] = useState<ReadonlySet<string>>(() => new Set());
  const [environment, setEnvironment] = useState(panelWorldEnvironmentFixture);

  const shaderAsset = useMemo(() => panelProjectFixture.assets.find((asset) => asset.kind === 'shader') ?? null, []);

  const selectEntity = (entityId: string, additive = false) => {
    setSelectedEntityId(entityId);
    setSelectedEntityIds((current) => {
      if (!additive) return new Set([entityId]);
      const next = new Set(current);
      if (next.has(entityId)) next.delete(entityId);
      else next.add(entityId);
      return next;
    });
  };

  const renderPanel = (id: WorkbenchPanelId) => {
    switch (id) {
      case 'viewport':
        return (
          <ViewportPanel
            active={false}
            onCommand={() => undefined}
            onReconnect={async () => undefined}
            project={panelProjectFixture}
            startupState={{
              appVersion: 'ui-lab',
              engineHostConnected: false,
              viewportMode: 'unavailable',
              hostError: 'UI Lab preview uses a static scene image instead of the native renderer.',
            }}
            viewportId="viewport-1"
          />
        );
      case 'hierarchy':
        return (
          <ExplorerPanel
            onCreateEntity={() => undefined}
            onCreatePrefab={() => undefined}
            onDelete={() => undefined}
            onDuplicate={() => undefined}
            onInstantiatePrefab={() => undefined}
            onMoveEntity={() => undefined}
            onRenameEntity={() => undefined}
            onSelectEntity={selectEntity}
            onSetEntityActive={() => undefined}
            project={panelProjectFixture}
            selectedEntityId={selectedEntityId}
            selectedEntityIds={selectedEntityIds}
          />
        );
      case 'inspector':
        return (
          <InspectorPanel
            assets={panelProjectFixture.assets}
            command={async () => ({ succeeded: true })}
            loading={false}
            onStatus={() => undefined}
            refresh={async () => undefined}
            snapshot={panelInspectorFixture}
            thumbnailProvider={async () => null}
          />
        );
      case 'assetExplorer':
        return <InternalPanelNotice id={id} />;
      case 'search':
        return (
          <SearchPanel
            assets={panelProjectFixture.assets}
            entities={panelProjectFixture.scene}
            onSelectAsset={setSelectedAssetId}
            onSelectEntity={(entityId) => selectEntity(entityId)}
          />
        );
      case 'renderGraph':
        return <RenderGraphPanel fixtureSnapshot={panelDiagnosticsFixture} queryHost={false} />;
      case 'shaderEditor':
        return <ShaderEditorPanel asset={shaderAsset} />;
      case 'lighting':
        return (
          <LightingPanel
            entities={panelProjectFixture.scene}
            fixtureDiagnostics={panelDiagnosticsFixture}
            onSelect={(entityId) => selectEntity(entityId)}
            queryHost={false}
          />
        );
      case 'worldSettings':
        return (
          <WorldEnvironmentInspector
            assets={panelProjectFixture.assets}
            environment={environment}
            onChange={setEnvironment}
            onHdri={() => true}
            onPreset={() => undefined}
            thumbnailProvider={async () => null}
          />
        );
      case 'contentBrowser':
        return (
          <ContentBrowserPanel
            cache={null}
            onAssetAction={() => undefined}
            onCommand={() => undefined}
            onInstantiatePrefab={() => undefined}
            onSelectAsset={setSelectedAssetId}
            project={panelProjectFixture}
            selectedAssetId={selectedAssetId}
            thumbnailProvider={async () => null}
          />
        );
      case 'console':
        return (
          <ConsolePanel
            clearedIds={clearedConsoleIds}
            events={panelProjectFixture.console}
            locked={consoleLocked}
            onClear={(events) => setClearedConsoleIds(new Set(events.map((event) => event.id)))}
            onLockedChange={setConsoleLocked}
          />
        );
      case 'buildOutput':
        return (
          <BuildOutputPanel
            onExecute={() => undefined}
            onOpenDiagnostic={() => undefined}
            snapshot={panelBuildFixture}
          />
        );
      case 'versionControl':
        return <VersionControlPanel />;
      case 'aiAssistant':
        return (
          <AiGatewayPanel
            onApprove={() => undefined}
            onCancelEdit={() => undefined}
            onDeny={() => undefined}
            onRevoke={() => undefined}
            onUndoLastEdit={() => undefined}
            status={panelGatewayFixture}
          />
        );
      case 'profiler':
        return <ProfilerPanel samples={panelProfilerFixtures} />;
      default:
        return <InternalPanelNotice id={id} />;
    }
  };

  return (
    <main className="ui-lab-panels-shell">
      <header className="ui-lab-panels-hero">
        <div>
          <strong>Panel Lab</strong>
          <span>Production editor panels mounted with deterministic fixture data.</span>
        </div>
        <div className="ui-lab-panels-meta">
          <span>{panelOrder.length} registered panels</span>
          <span>Real panel components</span>
          <span>Native renderer not required</span>
        </div>
      </header>

      <section className="ui-lab-panels-grid" aria-label="Editor panel gallery">
        {panelOrder.map((id) => (
          <PanelCard id={id} key={id}>
            {renderPanel(id)}
          </PanelCard>
        ))}
      </section>
    </main>
  );
}
