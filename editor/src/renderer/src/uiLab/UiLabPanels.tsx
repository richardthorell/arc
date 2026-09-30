import { useMemo, useState } from 'react';

import { ExplorerPanel } from '../app/Workbench';
import { panelRegistry } from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';
import { AiChatPanel } from '../ai/AiChatPanel';
import type { AiChatMessage, AiModelProvider } from '../ai/aiChat';
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
  aiAssistant: 'featured',
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
  aiAssistant: 'AiChatPanel ×2',
  profiler: 'ProfilerPanel',
};

const uiLabAiMessages: readonly AiChatMessage[] = [
  {
    id: 'ui-lab-user-1',
    role: 'user',
    content: 'Can you summarize the selected cabin and suggest one small polish pass?',
    createdAt: '2026-09-30T17:00:00Z',
    state: 'complete',
  },
  {
    id: 'ui-lab-agent-1',
    role: 'assistant',
    content:
      'The selected cabin is an asset-backed mesh with a material already assigned. A small polish pass could focus on the material response: reduce the roughness variation slightly, then check the result from both grazing and front-facing angles before changing any geometry.',
    createdAt: '2026-09-30T17:00:04Z',
    state: 'complete',
  },
];

const uiLabAiProvider: AiModelProvider = {
  id: 'ui-lab-mock',
  label: 'ARC Mock',
  configured: true,
  async *stream(request) {
    const prompt = [...request.messages].reverse().find((message) => message.role === 'user')?.content ?? 'that';
    const response = `Mock response received for “${prompt}”. This provider is intentionally deterministic so we can iterate on chat layout, streaming states, response cards, and future task cards without a live AI service.`;
    const chunks = response.match(/.{1,28}(?:\s|$)/g) ?? [response];
    for (const text of chunks) {
      yield { type: 'delta' as const, text };
      await Promise.resolve();
    }
    yield { type: 'done' as const };
  },
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
          <div className="ui-lab-ai-chat-variants">
            <section className="ui-lab-ai-chat-variant" aria-label="Disconnected AI Chat preview">
              <header>Disconnected</header>
              <AiChatPanel />
            </section>
            <section className="ui-lab-ai-chat-variant" aria-label="Mock provider AI Chat preview">
              <header>Mock provider</header>
              <AiChatPanel
                conversationLabel="Cabin polish"
                initialMessages={uiLabAiMessages}
                provider={uiLabAiProvider}
              />
            </section>
          </div>
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
