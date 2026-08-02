import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, FileCode2, Play, Save } from 'lucide-react';

import type { AssetItem } from '../services/editorHostTypes';
import type { HostResponse } from '../inspector/inspectorTypes';
import { UiButton } from '../ui';

import '../tools/tools.css';

const includePattern = /^\s*#\s*include\s*["<]([^">]+)[">]/gm;

export function ShaderEditorPanel({ asset }: { asset: AssetItem | null }) {
  const [source, setSource] = useState('');
  const [confirmed, setConfirmed] = useState('');
  const [modifiedAt, setModifiedAt] = useState('');
  const [message, setMessage] = useState('');
  const [compiling, setCompiling] = useState(false);
  const shader = asset?.kind === 'shader' ? asset : null;
  const shaderId = shader?.id;
  const shaderPath = shader?.path;
  const includes = useMemo(() => [...source.matchAll(includePattern)].map((match) => match[1]), [source]);
  const dirty = source !== confirmed;

  useEffect(() => {
    if (!shaderPath) {
      setSource('');
      setConfirmed('');
      return;
    }
    void window.arc.projects
      .readText(shaderPath)
      .then((file) => {
        setSource(file.text);
        setConfirmed(file.text);
        setModifiedAt(file.modifiedAt);
        setMessage('');
      })
      .catch((error) => setMessage(error instanceof Error ? error.message : String(error)));
  }, [shaderId, shaderPath]);

  const compile = async (save: boolean) => {
    if (!shader) return;
    setCompiling(true);
    try {
      if (save) {
        await window.arc.projects.writeText(shader.path, source);
        setConfirmed(source);
      }
      if (!shader.guid) {
        setMessage('This shader has no registered asset GUID and cannot be compiled');
        return;
      }
      const response = (await window.arc.host.command('asset.reimport', { guid: shader.guid })) as HostResponse;
      setMessage(
        response.succeeded
          ? save
            ? 'Shader saved and compilation queued'
            : 'Compilation queued from the last saved source'
          : response.error || 'Shader compilation failed',
      );
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setCompiling(false);
    }
  };

  if (!shader)
    return (
      <section className="production-tool-panel tool-empty-state">
        <FileCode2 size={29} />
        <strong>Select a shader asset</strong>
        <span>The shader editor only opens registered project shader sources.</span>
      </section>
    );

  return (
    <section className="production-tool-panel shader-editor-panel">
      <header className="tool-panel-toolbar">
        <FileCode2 size={15} />
        <strong>{shader.name}</strong>
        <span className="shader-dirty-state">
          {dirty ? 'Modified' : `Saved ${new Date(modifiedAt).toLocaleTimeString()}`}
        </span>
        <UiButton disabled={compiling} onClick={() => void compile(false)} variant="toolbar">
          <Play size={13} /> Compile
        </UiButton>
        <UiButton disabled={compiling || !dirty} onClick={() => void compile(true)} variant="primary">
          <Save size={13} /> Save & compile
        </UiButton>
      </header>
      <div className="shader-editor-body">
        <aside className="shader-include-tree">
          <h3>Include closure</h3>
          <button type="button">{shader.path}</button>
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
            {source.split('\n').map((_, index) => (
              <span key={index}>{index + 1}</span>
            ))}
          </div>
          <textarea
            aria-label="Shader source"
            spellCheck={false}
            value={source}
            onChange={(event) => setSource(event.target.value)}
          />
        </div>
        <aside className="shader-preview">
          <h3>Live preview</h3>
          <div className="shader-preview-sphere" />
          <p>Production PBR preview uses the last successfully published shader generation.</p>
        </aside>
      </div>
      {message && (
        <div className={message.toLocaleLowerCase().includes('failed') ? 'tool-error' : 'tool-message'}>
          <AlertCircle size={13} /> {message}
        </div>
      )}
    </section>
  );
}
