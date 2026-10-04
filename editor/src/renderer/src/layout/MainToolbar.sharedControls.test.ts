import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const layoutDirectory = dirname(fileURLToPath(import.meta.url));
const nativeToolbarControl = /<(?:button|input|select|textarea)\b/;

const readMainToolbar = () => readFileSync(resolve(layoutDirectory, 'MainToolbar.tsx'), 'utf8');

describe('MainToolbar shared controls', () => {
  it('keeps toolbar controls on shared ARC UI primitives', () => {
    const source = readMainToolbar();

    expect(source).not.toMatch(nativeToolbarControl);
    expect(source).toContain('UiIconButton');
    expect(source).toContain('UiDropdown');
    expect(source).toContain('UiSplitButton');
  });

  it('keeps the Scene toolbar on the shared region and group vocabulary', () => {
    const source = readMainToolbar();
    const left = source.indexOf('className="toolbar-left"');
    const center = source.indexOf('className="toolbar-center"');
    const right = source.indexOf('className="toolbar-right"');

    expect(left).toBeGreaterThan(-1);
    expect(center).toBeGreaterThan(left);
    expect(right).toBeGreaterThan(center);
    expect(source).toContain('ui-toolbar-group toolbar-group playback-group');
    expect(source).toContain('ui-toolbar-group toolbar-group');
    expect(source).toContain('className="toolbar-separator"');
  });
});
