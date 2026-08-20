import { useEffect, useRef } from 'react';
import type * as Monaco from 'monaco-editor/editor';

import { registerShaderLanguages, shaderLanguageForPath } from './language/ShaderLanguageRegistry';

import './ShaderSourceEditor.css';

type MonacoApi = typeof import('monaco-editor/editor');

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

let monacoPromise: Promise<MonacoApi> | null = null;

const loadMonaco = () => {
  if (!monacoPromise) {
    monacoPromise = Promise.all([
      import('monaco-editor/editor'),
      import('monaco-editor/editor/editor.worker?worker'),
      import('monaco-editor/features/register.all'),
    ]).then(([monaco, workerModule]) => {
      const globalWithMonaco = globalThis as typeof globalThis & {
        MonacoEnvironment?: MonacoEnvironment;
      };
      if (!globalWithMonaco.MonacoEnvironment) {
        globalWithMonaco.MonacoEnvironment = {
          getWorker: () => new workerModule.default(),
        };
      }
      registerShaderLanguages(monaco);
      return monaco;
    });
  }
  return monacoPromise;
};

const modelUri = (monaco: MonacoApi, documentId: string, path: string) =>
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
  const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);
  const modelRef = useRef<Monaco.editor.ITextModel | null>(null);
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
    let disposed = false;
    let cleanup: (() => void) | undefined;

    const initialize = async () => {
      const container = containerRef.current;
      if (!container) return;

      const monaco = await loadMonaco();
      if (disposed || !containerRef.current) return;

      const language = shaderLanguageForPath(path);
      const model = monaco.editor.createModel(valueRef.current, language.monacoId, modelUri(monaco, documentId, path));
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

      cleanup = () => {
        changeSubscription.dispose();
        editor.dispose();
        model.dispose();
        if (editorRef.current === editor) editorRef.current = null;
        if (modelRef.current === model) modelRef.current = null;
      };

      if (disposed) cleanup();
    };

    void initialize();

    return () => {
      disposed = true;
      cleanup?.();
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
