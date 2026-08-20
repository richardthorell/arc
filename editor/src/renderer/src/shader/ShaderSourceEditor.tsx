import { AlertCircle, FileCode2, Play, RefreshCw, Save } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { UiButton } from '../ui';
import {
  compileShaderDocument,
  reloadShaderDocument,
  saveAndCompileShaderDocument,
  saveShaderDocument,
  setShaderDocumentSource,
  useShaderDocumentState,
} from './shaderDocumentState';

import '../tools/tools.css';

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
        disabled={state.compiling || state.loading || !document.assetGuid}
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
            {dirty
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
        <aside className="shader-include-tree">
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
        </aside>
        <div className="shader-source-editor">
          <div className="shader-source-gutter" aria-hidden="true">
            {state.source.split('\n').map((_, index) => (
              <span key={index}>{index + 1}</span>
            ))}
          </div>
          <textarea
            aria-label="Shader source"
            disabled={state.loading || document.readOnly}
            spellCheck={false}
            value={state.source}
            onChange={(event) => setShaderDocumentSource(document, event.target.value)}
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === 's') {
                event.preventDefault();
                void saveShaderDocument(document);
              }
            }}
          />
        </div>
        <aside className="shader-preview">
          <h3>Live preview</h3>
          <div className="shader-preview-sphere" />
          <p>Production PBR preview uses the last successfully published shader generation.</p>
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
