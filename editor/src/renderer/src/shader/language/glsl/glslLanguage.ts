import type { ShaderLanguageDefinition } from '../ShaderLanguage';
import { glslSymbols } from './glslBuiltins';
import { glslTokenizer } from './glslTokenizer';

export const glslLanguageDefinition: ShaderLanguageDefinition = {
  id: 'glsl',
  monacoId: 'arc-glsl',
  aliases: ['GLSL', 'glsl'],
  extensions: ['.vert', '.frag', '.geom', '.tesc', '.tese', '.comp', '.glsl'],
  configuration: {
    comments: {
      lineComment: '//',
      blockComment: ['/*', '*/'],
    },
    brackets: [
      ['{', '}'],
      ['[', ']'],
      ['(', ')'],
    ],
    autoClosingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"', notIn: ['string', 'comment'] },
    ],
    surroundingPairs: [
      { open: '{', close: '}' },
      { open: '[', close: ']' },
      { open: '(', close: ')' },
      { open: '"', close: '"' },
    ],
  },
  tokenizer: glslTokenizer,
  symbols: glslSymbols,
};
