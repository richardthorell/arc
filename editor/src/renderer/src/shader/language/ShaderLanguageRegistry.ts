import type * as Monaco from 'monaco-editor/editor';

import type { ShaderLanguageDefinition, ShaderSymbol } from './ShaderLanguage';
import { glslLanguageDefinition } from './glsl/glslLanguage';

export type MonacoApi = typeof import('monaco-editor/editor');

const definitions: readonly ShaderLanguageDefinition[] = [glslLanguageDefinition];
const symbolIndexes = new Map(
  definitions.map((definition) => [
    definition.monacoId,
    new Map(definition.symbols.map((symbol) => [symbol.name, symbol])),
  ]),
);

let registered = false;

const symbolKindLabel = (symbol: ShaderSymbol) => {
  switch (symbol.kind) {
    case 'function':
      return 'GLSL built-in function';
    case 'type':
      return 'GLSL built-in type';
    case 'variable':
      return 'GLSL built-in variable';
  }
};

const hoverContents = (symbol: ShaderSymbol): Monaco.IMarkdownString[] => {
  const contents: Monaco.IMarkdownString[] = [{ value: `**${symbol.name}**` }];
  for (const signature of symbol.signatures ?? []) {
    contents.push({ value: `\`\`\`glsl\n${signature.label}\n\`\`\`` });
    if (signature.documentation) contents.push({ value: signature.documentation });
  }
  contents.push({ value: symbol.description }, { value: `_${symbolKindLabel(symbol)}_` });
  return contents;
};

export const shaderLanguageForPath = (path: string): ShaderLanguageDefinition => {
  const lowerPath = path.toLocaleLowerCase();
  return (
    definitions.find((definition) => definition.extensions.some((extension) => lowerPath.endsWith(extension))) ??
    glslLanguageDefinition
  );
};

export const getShaderSymbol = (monacoLanguageId: string, name: string): ShaderSymbol | undefined =>
  symbolIndexes.get(monacoLanguageId)?.get(name);

export const registerShaderLanguages = (monaco: MonacoApi) => {
  if (registered) return;
  registered = true;

  for (const definition of definitions) {
    monaco.languages.register({
      id: definition.monacoId,
      aliases: [...definition.aliases],
      extensions: [...definition.extensions],
    });
    monaco.languages.setLanguageConfiguration(definition.monacoId, definition.configuration);
    monaco.languages.setMonarchTokensProvider(definition.monacoId, definition.tokenizer);
    monaco.languages.registerHoverProvider(definition.monacoId, {
      provideHover(model, position) {
        const word = model.getWordAtPosition(position);
        if (!word) return null;
        const symbol = getShaderSymbol(definition.monacoId, word.word);
        if (!symbol) return null;

        return {
          range: new monaco.Range(position.lineNumber, word.startColumn, position.lineNumber, word.endColumn),
          contents: hoverContents(symbol),
        };
      },
    });
  }

  monaco.editor.defineTheme('arc-shader-dark', {
    base: 'vs-dark',
    inherit: true,
    rules: [
      { token: 'keyword.glsl', foreground: 'C586C0' },
      { token: 'keyword.directive.glsl', foreground: 'C586C0' },
      { token: 'type.glsl', foreground: '4EC9B0' },
      { token: 'predefined.glsl', foreground: 'DCDCAA' },
      { token: 'variable.predefined.glsl', foreground: '9CDCFE' },
      { token: 'number.glsl', foreground: 'B5CEA8' },
      { token: 'number.float.glsl', foreground: 'B5CEA8' },
      { token: 'number.hex.glsl', foreground: 'B5CEA8' },
      { token: 'comment.glsl', foreground: '6A9955' },
      { token: 'string.glsl', foreground: 'CE9178' },
    ],
    colors: {
      'editor.background': '#0f171c',
      'editor.foreground': '#cfdae1',
      'editor.lineHighlightBackground': '#16242c',
      'editor.selectionBackground': '#1b6175',
      'editor.inactiveSelectionBackground': '#24434f',
      'editorLineNumber.foreground': '#5c7180',
      'editorLineNumber.activeForeground': '#d8f3f5',
      'editorCursor.foreground': '#d8f3f5',
      'editorIndentGuide.background1': '#26343d',
      'editorIndentGuide.activeBackground1': '#3a4c57',
      'editorHoverWidget.background': '#18242b',
      'editorHoverWidget.border': '#34444e',
    },
  });
};
