import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const layoutDirectory = dirname(fileURLToPath(import.meta.url));
const nativeToolbarControl = /<(?:button|input|select|textarea)\b/;

describe('MainToolbar shared controls', () => {
  it('keeps toolbar controls on shared ARC UI primitives', () => {
    const source = readFileSync(resolve(layoutDirectory, 'MainToolbar.tsx'), 'utf8');

    expect(source).not.toMatch(nativeToolbarControl);
    expect(source).toContain('UiIconButton');
    expect(source).toContain('UiDropdown');
    expect(source).toContain('UiSplitButton');
  });
});
