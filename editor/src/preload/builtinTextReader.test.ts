import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { readBuiltinTextFile, resolveBuiltinAssetsRoot } from './builtinTextReader';

const temporaryRoots: string[] = [];
const temporary = () => {
  const value = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-builtin-text-'));
  temporaryRoots.push(value);
  return value;
};

afterEach(() => {
  for (const root of temporaryRoots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('built-in text reader', () => {
  it('loads builtin paths from the engine assets root', () => {
    const root = temporary();
    const assets = path.join(root, 'assets');
    const shader = path.join(assets, 'shaders', 'default_unlit.frag');
    fs.mkdirSync(path.dirname(shader), { recursive: true });
    fs.writeFileSync(shader, '#version 450\nvoid main() {}\n', 'utf8');

    expect(resolveBuiltinAssetsRoot({ environmentRoot: assets, cwd: root })).toBe(assets);
    expect(readBuiltinTextFile('builtin/shaders/default_unlit.frag', { environmentRoot: assets, cwd: root })).toMatchObject({
      path: 'builtin/shaders/default_unlit.frag',
      text: '#version 450\nvoid main() {}\n',
    });
  });

  it('rejects paths that escape the built-in asset root', () => {
    const root = temporary();
    const assets = path.join(root, 'assets');
    fs.mkdirSync(assets, { recursive: true });

    expect(() => readBuiltinTextFile('builtin/../secret.txt', { environmentRoot: assets, cwd: root })).toThrow(
      'relative to the engine asset root',
    );
  });
});
