import { useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, CircleAlert, Code2, Eye, LoaderCircle, RefreshCw, Save, Upload, Zap } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import {
  UiButton,
  UiContextMenu,
  UiContextMenuItem,
  UiEditorToolbar,
  UiSplitButton,
  UiToggleButton,
  UiToolbarSeparator,
} from '../ui';
import {
  compileMaterialDocument,
  reloadMaterialDocument,
  saveAndPublishMaterialDocument,
  saveMaterialDocument,
  setMaterialGraphView,
  setMaterialLiveUpdate,
  useMaterialDocumentState,
} from './materialDocumentState';

import '../tools/tools.css';

const menuCheck = (checked: boolean) =>
  checked ? <Check size={13} /> : <span aria-hidden="true" className="material-toolbar-check-placeholder" />;

export function MaterialEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useMaterialDocumentState(document);
  const toolbarRef = useRef<HTMLDivElement | null>(null);
  const [viewOpen, setViewOpen] = useState(false);
  const customShader = typeof state.asset.shaderPath === 'string' && state.asset.shaderPath.trim().length > 0;
  const busy = state.loading || state.saving || state.compiling;
  const errorCount = state.compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'error').length;

  useEffect(() => {
    if (!viewOpen) return;
    const close = (event: PointerEvent) => {
      if (!toolbarRef.current?.contains(event.target as Node)) setViewOpen(false);
    };
    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, [viewOpen]);

  const compilePresentation = customShader
    ? { label: 'Reimport', icon: <Code2 size={13} />, tone: 'idle' }
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

  return (
    <UiEditorToolbar
      aria-label="Material editor toolbar"
      className="material-document-toolbar"
      ref={toolbarRef}
      left={
        <>
          <UiButton
            disabled={busy || document.readOnly || !document.dirty}
            onClick={() => void saveMaterialDocument(document)}
            variant="toolbar"
          >
            <Save size={13} /> Save
          </UiButton>

          <UiSplitButton
            ariaLabel={compilePresentation.label}
            className={`material-toolbar-compile material-toolbar-compile-${compilePresentation.tone}`}
            disabled={state.loading || state.compiling}
            icon={compilePresentation.icon}
            label={compilePresentation.label}
            menuAriaLabel="Material compile actions"
            onClick={() =>
              void (customShader ? saveAndPublishMaterialDocument(document) : compileMaterialDocument(document))
            }
            onOptionSelect={(value) => {
              if (value === 'publish') void saveAndPublishMaterialDocument(document);
              else void reloadMaterialDocument(document);
            }}
            options={[
              {
                value: 'publish',
                label: customShader ? 'Save & Reimport' : 'Save & Compile',
                icon: <Upload size={13} />,
                disabled: busy || document.readOnly || !document.assetGuid,
              },
              {
                value: 'reload',
                label: 'Reload from Disk',
                icon: <RefreshCw size={13} />,
                disabled: busy,
              },
            ]}
            variant="toolbar"
          />

          <UiToolbarSeparator />

          {!customShader && (
            <UiToggleButton
              aria-label="Live Update"
              checked={state.liveUpdate}
              className="material-toolbar-live-toggle"
              label="Live Update"
              onCheckedChange={(enabled) => setMaterialLiveUpdate(document, enabled)}
            />
          )}

          {!customShader && (
            <span className="material-toolbar-menu material-toolbar-view-menu">
              <UiButton
                aria-expanded={viewOpen}
                aria-haspopup="menu"
                onClick={() => setViewOpen((open) => !open)}
                variant="toolbar"
              >
                <Eye size={13} /> View <ChevronDown aria-hidden="true" size={12} />
              </UiButton>
              {viewOpen && (
                <UiContextMenu aria-label="Material graph view options" className="material-toolbar-popup" width={210}>
                  <UiContextMenuItem
                    leading={menuCheck(state.showGrid)}
                    onClick={() => {
                      setViewOpen(false);
                      setMaterialGraphView(document, { showGrid: !state.showGrid });
                    }}
                  >
                    Show Grid
                  </UiContextMenuItem>
                  <UiContextMenuItem
                    leading={menuCheck(state.dimUnrelated)}
                    onClick={() => {
                      setViewOpen(false);
                      setMaterialGraphView(document, { dimUnrelated: !state.dimUnrelated });
                    }}
                  >
                    Dim Unrelated
                  </UiContextMenuItem>
                  <UiContextMenuItem
                    leading={menuCheck(state.showStats)}
                    onClick={() => {
                      setViewOpen(false);
                      setMaterialGraphView(document, { showStats: !state.showStats });
                    }}
                  >
                    Stats Overlay
                  </UiContextMenuItem>
                </UiContextMenu>
              )}
            </span>
          )}
        </>
      }
    />
  );
}
