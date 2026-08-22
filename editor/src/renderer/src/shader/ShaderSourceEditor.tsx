import { AlertCircle, FileCode2, Play, RefreshCw, Save } from 'lucide-react';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import { AssetPreviewViewport } from '../assetPreview/AssetPreviewViewport';
import type { EditorDocument } from '../editors/editorTypes';
import { UiButton } from '../ui';
import { ShaderCodeEditor } from './ShaderCodeEditor';
import {
  compileShaderDocument,
  reloadShaderDocument,
  saveAndCompileShaderDocument,
  saveShaderDocument,
  setShaderDocumentSource,
  useShaderDocumentState,
} from './shaderDocumentState';

import '../tools/tools.css';
import './ShaderSourceEditor.css';

const includePattern = /^\s*#\s*include\s*["<]([^">]+)[">]/gm;

export function ShaderEditorActions({ document }: { document: EditorDocument }) {
  const state = useShaderDocumentState(document);
  const dirty = state.source !== state.confirmed;

  return (
    <>
      <UiButton
        disabled={state.compiling || state.loading || document.readOnly || !dirty}
        onClick={() => void saveShaderDocument(document)}
        variant="toolbar"
      >
        <Save size={13} /> Save
      </UiButton>
      <UiButton
        disabled={state.compiling || state.loading || document.readOnly || !document.assetGuid}
        onClick={() => void compileShaderDocument(document)}
        variant="toolbar"
      >
        <Play size={13} /> Compile
      </UiButton>
      <UiButton
        disabled={state.compiling || state.loading || document.readOnly || !dirty || !document.assetGuid}
        onClick={() => void saveAndCompileShaderDocument(document)}
        variant="primary"
      >
        <Save size={13} /> Save &amp; Compile
      </UiButton>
      <UiButton
        disabled={state.compiling || state.loading}
        onClick={() => void reloadShaderDocument(document)}
        variant="toolbar"
      >
        <RefreshCw size={13} /> Reload
      </UiButton>
    </>
  );
}

export function ShaderSourceEditor({
  document,
  embeddedToolbar = false,
}: {
  document: EditorDocument;
  embeddedToolbar?: boolean;
}) {
  const state = useShaderDocumentState(document);
  const includes = [...state.source.matchAll(includePattern)].map((match) => match[1]);
  const dirty = state.source !== state.confirmed;

  return (
    <section className="production-tool-panel shader-editor-panel shader-document-editor">
      {embeddedToolbar && (
        <header className="tool-panel-toolbar">
          <FileCode2 size={15} />
          <strong>{document.title}</strong>
          <span className="shader-dirty-state">
            {document.readOnly
              ? 'Read-only'
              : dirty
                ? 'Modified'
                : state.modifiedAt
                  ? `Saved ${new Date(state.modifiedAt).toLocaleTimeString()}`
                  : state.loading
                    ? 'Loading...'
                    : 'Saved'}
          </span>
          <ShaderEditorActions document={document} />
        </header>
      )}
      <div className="shader-editor-body">
        <div className="shader-source-editor">
          <ShaderCodeEditor
            documentId={document.id}
            path={document.path ?? document.title}
            value={state.source}
            readOnly={document.readOnly}
            loading={state.loading}
            onChange={(source) => setShaderDocumentSource(document, source)}
            onSave={() => void saveShaderDocument(document)}
          />
        </div>
        <aside className="shader-side-panel">
          <AssetPreviewPanel
            title="Shader Preview"
            subtitle="Native renderer"
            metadata={[
              { label: 'Mesh', value: 'Sphere' },
              { label: 'Environment', value: 'Studio' },
            ]}
          >
            <AssetPreviewViewport
              kind="shader"
              assetGuid={document.assetGuid}
              label={`${document.title} shader preview viewport`}
              fallback={
                <AssetPreviewPlaceholder
                  label="Shader preview"
                  description="A registered shader asset is required for the native preview surface."
                />
              }
            />
          </AssetPreviewPanel>
          <section className="shader-include-tree">
            <h3>Include closure</h3>
            <button type="button">{document.path}</button>
            {includes.map((include) => (
              <button key={include} type="button">
                {include}
              </button>
            ))}
            {!includes.length && <p>No direct includes.</p>}
            <h3>Permutation</h3>
            <label>
              Entry point
              <select defaultValue="main">
                <option>main</option>
              </select>
            </label>
            <label>
              Target
              <select defaultValue="spirv">
                <option value="spirv">SPIR-V · Vulkan 1.2</option>
              </select>
            </label>
          </section>
        </aside>
      </div>
      {state.message && (
        <div className={state.message.toLocaleLowerCase().includes('failed') ? 'tool-error' : 'tool-message'}>
          <AlertCircle size={13} /> {state.message}
        </div>
      )}
    </section>
  );
}
