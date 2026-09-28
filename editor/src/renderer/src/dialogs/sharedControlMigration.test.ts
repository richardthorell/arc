import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const nativeControlPattern = /<(?:button|input|select|textarea)\b/i;

describe('dialog shared control migration', () => {
  it('keeps ImportDialog on shared Ui controls', () => {
    const source = readFileSync(fileURLToPath(new URL('ImportDialog.tsx', import.meta.url)), 'utf8');

    expect(source).not.toMatch(nativeControlPattern);
    expect(source).toContain('UiTextInput');
    expect(source).toContain('UiToggleButton');
    expect(source).toContain('UiSelect');
  });
});
