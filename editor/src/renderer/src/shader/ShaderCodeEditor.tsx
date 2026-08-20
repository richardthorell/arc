import { useEffect, useRef } from 'react';
import * as monaco from 'monaco-editor/editor';
import 'monaco-editor/features/register.all';
import EditorWorker from 'monaco-editor/editor/editor.worker?worker';

import { registerShaderLanguages, shaderLanguageForPath } from './language/ShaderLanguageRegistry';

import './ShaderSourceEditor.css';

type MonacoEnvironment = {
  getWorker: (_moduleId: string, _label: string) => Worker;
};

type ShaderCodeEditorProps = {
  documentId: string;
  path: string;
  value: string;
  readOnly: boolean;
  loading: boolean;
  onChange: (value: string) => void;
  onSave: () => void;
};

const globalWithMonaco = globalThis as typeof globalThis & {
  MonacoEnvironment?: MonacoEnvironment;
};

if (!globalWithMonaco.MonacoEnvironment) {
  globalWithMonaco.MonacoEnvironment = {
    getWorker: () => new EditorWorker(),
  };
}

registerShaderLanguages(monaco);

const modelUri = (documentId: string, path: string) =>
  monaco.Uri.from({
    scheme: 'arc-shader',
    path: `/${encodeURIComponent(documentId)}/${path.replace(/^\/+/, '')}`,
  });

export function ShaderCodeEditor({
  documentId,
  path,
  value,
  readOnly,
  loading,
  onChange,
  onSave,
}: ShaderCodeEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);
  const modelRef = useRef<monaco.editor.ITextModel | null>(null);
  const applyingExternalValueRef = useRef(false);
  const valueRef = useRef(value);
  const readOnlyRef = useRef(readOnly);
  const loadingRef = useRef(loading);
  const onChangeRef = useRef(onChange);
  const onSaveRef = useRef(onSave);

  valueRef.current = value;
  readOnlyRef.current = readOnly;
  loadingRef.current = loading;
  onChangeRef.current = onChange;
  onSaveRef.current = onSave;

  useEffect(() => {
    if (!containerRef.current) return;

    const language = shaderLanguageForPath(path);
    const model = monaco.editor.createModel(valueRef.current, language.monacoId, modelUri(documentId, path));
    const editor = monaco.editor.create(containerRef.current, {
      model,
      theme: 'arc-shader-dark',
      readOnly: readOnlyRef.current || loadingRef.current,
      automaticLayout: true,
      fontFamily: "Consolas, 'Cascadia Code', monospace",
      fontSize: 12,
      lineHeight: 19,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      renderWhitespace: 'selection',
      bracketPairColorization: { enabled: true },
      guides: { bracketPairs: true, indentation: true },
      padding: { top: 8, bottom: 8 },
      tabSize: 4,
      insertSpaces: true,
      wordWrap: 'off',
      fixedOverflowWidgets: true,
      ariaLabel: 'Shader source',
    });

    editorRef.current = editor;
    modelRef.current = model;

    const changeSubscription = model.onDidChangeContent(() => {
      if (!applyingExternalValueRef.current) onChangeRef.current(model.getValue());
    });
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => onSaveRef.current());

    return () => {
      changeSubscription.dispose();
      editor.dispose();
      model.dispose();
      editorRef.current = null;
      modelRef.current = null;
    };
  }, [documentId, path]);

  useEffect(() => {
    const model = modelRef.current;
    if (!model || model.getValue() === value) return;
    applyingExternalValueRef.current = true;
    model.setValue(value);
    applyingExternalValueRef.current = false;
  }, [value]);

  useEffect(() => {
    editorRef.current?.updateOptions({ readOnly: readOnly || loading });
  }, [loading, readOnly]);

  return <div ref={containerRef} className={`shader-code-editor${loading ? ' is-loading' : ''}`} />;
}
