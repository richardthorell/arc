import { useEffect, useMemo, useState, type ComponentProps, type SyntheticEvent } from 'react';
import { Lock } from 'lucide-react';

import type { AssetItem, ProjectSnapshot } from '../services/editorHostTypes';
import { assetLibraryIdentity } from './assetLibraryIdentity';
import { assetLibraryScopeViewsForProject } from './assetLibraryScopeSnapshot';
import type { AssetLibraryScopeId } from './assetLibraryScopes';
import { ContentBrowserPanel as ContentBrowserPanelCore } from './ContentBrowserPanelCore';

import './contentBrowserScopes.css';

export { buildContentFolderTree } from './ContentBrowserPanelCore';

type Props = ComponentProps<typeof ContentBrowserPanelCore>;

const logicalRootForScope = (project: ProjectSnapshot, scope: AssetLibraryScopeId): string => {
  switch (scope) {
    case 'builtin':
      return 'Engine';
    case 'user':
      return 'User';
    case 'organization':
      return 'Organization';
    case 'project':
      return project.assetRoot;
  }
};

const projectForScope = (
  project: ProjectSnapshot,
  scope: AssetLibraryScopeId,
  assetIds: readonly string[],
): ProjectSnapshot => {
  const visibleIds = new Set(assetIds);
  return {
    ...project,
    assetRoot: logicalRootForScope(project, scope),
    assets: project.assets.map((asset) =>
      visibleIds.has(assetLibraryIdentity(asset))
        ? ({ ...asset, scope: 'project' } as AssetItem)
        : ({ ...asset, scope: 'organization' } as AssetItem),
    ),
  };
};

export function ContentBrowserPanel(props: Props) {
  const scopeViews = useMemo(() => assetLibraryScopeViewsForProject(props.project), [props.project]);
  const availableScopes = useMemo(() => scopeViews.filter((scope) => scope.available), [scopeViews]);
  const [scopeId, setScopeId] = useState<AssetLibraryScopeId>('project');

  useEffect(() => {
    if (availableScopes.some((scope) => scope.scope === scopeId)) return;
    setScopeId(
      availableScopes.find((scope) => scope.scope === 'project')?.scope ?? availableScopes[0]?.scope ?? 'project',
    );
  }, [availableScopes, scopeId]);

  const activeScope =
    availableScopes.find((scope) => scope.scope === scopeId) ??
    availableScopes.find((scope) => scope.scope === 'project') ??
    null;
  const scopedProject =
    props.project && activeScope
      ? projectForScope(props.project, activeScope.scope, activeScope.assetIds)
      : props.project;
  const projectAuthoring = activeScope?.scope === 'project' && activeScope.writable;

  const suppressNonProjectMutation = (event: SyntheticEvent) => {
    if (projectAuthoring) return;
    event.preventDefault();
    event.stopPropagation();
  };

  return (
    <div
      className={`content-browser-scope-shell ${projectAuthoring ? '' : 'scope-authoring-disabled'}`}
      onContextMenuCapture={(event) => {
        if (!projectAuthoring && !(event.target as HTMLElement).closest('.content-asset'))
          suppressNonProjectMutation(event);
      }}
      onDragOverCapture={(event) => {
        if (!projectAuthoring) suppressNonProjectMutation(event);
      }}
      onDropCapture={(event) => {
        if (!projectAuthoring) suppressNonProjectMutation(event);
      }}
    >
      {availableScopes.length > 0 && (
        <nav className="content-browser-scope-nav" aria-label="Asset library scopes">
          {availableScopes.map((scope) => {
            const selected = scope.scope === activeScope?.scope;
            return (
              <button
                className={`content-browser-scope-button ${selected ? 'active' : ''}`}
                key={scope.scope}
                type="button"
                aria-pressed={selected}
                title={scope.description}
                onClick={() => setScopeId(scope.scope)}
              >
                {!scope.writable && <Lock size={11} aria-hidden="true" />}
                <span>{scope.label}</span>
                <span className={`content-browser-scope-access ${scope.writable ? 'writable' : 'read-only'}`}>
                  {scope.writable ? 'Writable' : 'Read only'}
                </span>
              </button>
            );
          })}
        </nav>
      )}
      <div className="content-browser-scope-core">
        <ContentBrowserPanelCore {...props} project={scopedProject} />
      </div>
    </div>
  );
}
