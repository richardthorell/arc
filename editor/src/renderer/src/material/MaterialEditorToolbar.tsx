import { useEffect, useRef, useState } from 'react';
import {
  BarChart3,
  Check,
  ChevronDown,
  CircleAlert,
  Code2,
  Ellipsis,
  Eye,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  RotateCw,
  Save,
  Upload,
  Zap,
} from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiContextMenu, UiContextMenuItem } from '../ui';
import {
  compileMaterialDocument,
  redoMaterialGraph,
  reloadMaterialDocument,
  saveAndPublishMaterialDocument,
  saveMaterialDocument,
  setMaterialGraphView,
  setMaterialLiveUpdate,
  undoMaterialGraph,
  useMaterialDocumentState,
} from './materialDocumentState';

import '../tools/tools.css';

type MaterialToolbarMenu = 'live' | 'stats' | 'view' | 'more';

const menuCheck = (checked: boolean) =>
  checked ? <Check size={13} /> : <span aria-hidden="true" className="material-toolbar-check-placeholder" />;

export function MaterialEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useMaterialDocumentState(document);
  const toolbarRef = useRef<HTMLDivElement | null>(null);
  const [openMenu, setOpenMenu] = useState<MaterialToolbarMenu | null>(null);
  const customShader = typeof state.asset.shaderPath === 'string' && state.asset.shaderPath.trim().length > 0;
  const busy = state.loading || state.saving || state.compiling;
  const canUndo = !customShader && !document.readOnly && state.historyIndex > 0;
  const canRedo = !customShader && !document.readOnly && state.historyIndex + 1 < state.history.length;
  const errorCount = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'error').length;
  const warningCount = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'warning').length;
  const parameterCount = state.graph.nodes.filter((node) => node.parameter?.exposed).length;

  useEffect(() => {
    if (!openMenu) return;
    const close = (event: PointerEvent) => {
      if (!toolbarRef.current?.contains(event.target as Node)) setOpenMenu(null);
    };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [openMenu]);

  const toggleMenu = (menu: MaterialToolbarMenu) =>
    setOpenMenu((current) => (current === menu ? null : menu));

  const compilePresentation = customShader
    ? { label: 'Cook validation', icon: <Code2 size={13} />, tone: 'idle' }
    : state.compilation.status === 'compiling'
      ? { label: 'Compiling…', icon: <LoaderCircle className="spinning" size={13} />, tone: 'busy' }
      : state.compilation.status === 'failed'
        ? {
            label: errorCount === 1 ? '1 error' : `${Math.max(1, errorCount)} errors`,
            icon: <CircleAlert size={13} />,
            tone: 'error',
          }
        : state.compilation.succeeded
          ? { label: 'Compiled', icon: <Check size={13} />, tone: 'success' }
          : { label: 'Compile', icon: <Zap size={13} />, tone: 'idle' };

  const closeAnd = (action: () => void) => {
    setOpenMenu(null);
    action();
  };

  return (
    <div aria-label="Material editor toolbar" className="main-toolbar material-document-toolbar" ref={toolbarRef}>
      <div className="toolbar-left">
        <UiButton
          disabled={busy || document.readOnly || !document.dirty}
          onClick={() => void saveMaterialDocument(document)}
          variant="toolbar"
        >
          <Save size={13} /> Save
        </UiButton>

        {!customShader && (
          <UiButton
            className={`material-toolbar-compile material-toolbar-compile-${compilePresentation.tone}`}
            disabled={state.loading || state.compiling}
            onClick={() => void compileMaterialDocument(document)}
            title={
              state.compilation.status === 'failed'
                ? state.compilation.diagnostics[0]?.message || 'Material compilation failed'
                : 'Compile the current in-memory material graph'
            }
            variant="toolbar"
          >
            {compilePresentation.icon}
            {compilePresentation.label}
          </UiButton>
        )}
        {customShader && (
          <UiButton className="material-toolbar-compile material-toolbar-compile-idle" disabled variant="toolbar">
            {compilePresentation.icon}
            {compilePresentation.label}
          </UiButton>
        )}

        <span className="toolbar-separator" />

        <span className="material-toolbar-menu">
          <UiButton
            aria-expanded={openMenu === 'live'}
            aria-haspopup="menu"
            aria-pressed={state.liveUpdate}
            className={state.liveUpdate ? 'is-active' : undefined}
            disabled={customShader}
            onClick={() => toggleMenu('live')}
            variant="toolbar"
          >
            <Zap size={13} /> Live Update <ChevronDown size={12} />
          </UiButton>
          {openMenu === 'live' && (
            <UiContextMenu aria-label="Live update options" className="material-toolbar-popup" width={190}>
              <UiContextMenuItem
                leading={menuCheck(state.liveUpdate)}
                onClick={() => closeAnd(() => setMaterialLiveUpdate(document, true))}
              >
                Enabled
              </UiContextMenuItem>
              <UiContextMenuItem
                leading={menuCheck(!state.liveUpdate)}
                onClick={() => closeAnd(() => setMaterialLiveUpdate(document, false))}
              >
                Paused
              </UiContextMenuItem>
            </UiContextMenu>
          )}
        </span>

        <span className="material-toolbar-menu">
          <UiButton
            aria-expanded={openMenu === 'stats'}
            aria-haspopup="dialog"
            onClick={() => toggleMenu('stats')}
            variant="toolbar"
          >
            <BarChart3 size={13} /> Stats
          </UiButton>
          {openMenu === 'stats' && (
            <div aria-label="Material stats" className="material-toolbar-stats menu-dropdown" role="dialog">
              <div>
                <span>Nodes</span>
                <strong>{customShader ? '—' : state.graph.nodes.length}</strong>
              </div>
              <div>
                <span>Connections</span>
                <strong>{customShader ? '—' : state.graph.connections.length}</strong>
              </div>
              <div>
                <span>Parameters</span>
                <strong>{customShader ? '—' : parameterCount}</strong>
              </div>
              <div>
                <span>Diagnostics</span>
                <strong>
                  {errorCount} error{errorCount === 1 ? '' : 's'} · {warningCount} warning
                  {warningCount === 1 ? '' : 's'}
                </strong>
              </div>
            </div>
          )}
        </span>

        {!customShader && (
          <span className="material-toolbar-menu">
            <UiButton
              aria-expanded={openMenu === 'view'}
              aria-haspopup="menu"
              onClick={() => toggleMenu('view')}
              variant="toolbar"
            >
              <Eye size={13} /> View <ChevronDown size={12} />
            </UiButton>
            {openMenu === 'view' && (
              <UiContextMenu aria-label="Material graph view options" className="material-toolbar-popup" width={210}>
                <UiContextMenuItem
                  leading={menuCheck(state.showGrid)}
                  onClick={() => closeAnd(() => setMaterialGraphView(document, { showGrid: !state.showGrid }))}
                >
                  Show Grid
                </UiContextMenuItem>
                <UiContextMenuItem
                  leading={menuCheck(state.dimUnrelated)}
                  onClick={() => closeAnd(() => setMaterialGraphView(document, { dimUnrelated: !state.dimUnrelated }))}
                >
                  Dim Unrelated
                </UiContextMenuItem>
              </UiContextMenu>
            )}
          </span>
        )}

        <span className="material-toolbar-menu material-toolbar-menu-more">
          <UiButton
            aria-expanded={openMenu === 'more'}
            aria-haspopup="menu"
            aria-label="More material actions"
            onClick={() => toggleMenu('more')}
            variant="toolbar"
          >
            <Ellipsis size={15} /> More <ChevronDown size={12} />
          </UiButton>
          {openMenu === 'more' && (
            <UiContextMenu
              aria-label="More material actions"
              className="material-toolbar-popup material-toolbar-popup-right"
              width={220}
            >
              <UiContextMenuItem
                disabled={busy || document.readOnly || !document.assetGuid}
                leading={<Upload size={13} />}
                onClick={() => closeAnd(() => void saveAndPublishMaterialDocument(document))}
              >
                {customShader ? 'Save & Reimport' : 'Save & Compile'}
              </UiContextMenuItem>
              <UiContextMenuItem
                disabled={busy}
                leading={<RefreshCw size={13} />}
                onClick={() => closeAnd(() => void reloadMaterialDocument(document))}
              >
                Reload
              </UiContextMenuItem>
              {!customShader && (
                <>
                  <UiContextMenuItem
                    disabled={!canUndo}
                    leading={<RotateCcw size={13} />}
                    onClick={() => closeAnd(() => void undoMaterialGraph(document))}
                    trailing="Ctrl+Z"
                  >
                    Undo
                  </UiContextMenuItem>
                  <UiContextMenuItem
                    disabled={!canRedo}
                    leading={<RotateCw size={13} />}
                    onClick={() => closeAnd(() => void redoMaterialGraph(document))}
                    trailing="Ctrl+Y"
                  >
                    Redo
                  </UiContextMenuItem>
                </>
              )}
            </UiContextMenu>
          )}
        </span>
      </div>
    </div>
  );
}
