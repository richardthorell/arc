import type * as Monaco from 'monaco-editor/editor';

export type ShaderSymbolKind = 'type' | 'function' | 'variable';

export type ShaderSignature = {
  label: string;
  documentation?: string;
};

export type ShaderSymbol = {
  name: string;
  kind: ShaderSymbolKind;
  description: string;
  signatures?: readonly ShaderSignature[];
};

export type ShaderLanguageDefinition = {
  id: string;
  monacoId: string;
  aliases: readonly string[];
  extensions: readonly string[];
  configuration: Monaco.languages.LanguageConfiguration;
  tokenizer: Monaco.languages.IMonarchLanguage;
  symbols: readonly ShaderSymbol[];
};
