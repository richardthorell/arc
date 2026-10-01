import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const migratedInspectorSurfaces = ['ColorPicker.tsx', 'InspectorComponentCard.tsx', 'InspectorControls.tsx'] as const;

const nativeControlPattern = /<(?:button|input|select|textarea)\b/i;

describe('inspector shared control migration', () => {
  it.each(migratedInspectorSurfaces)('%s stays on shared Ui controls', (filename) => {
    const source = readFileSync(fileURLToPath(new URL(filename, import.meta.url)), 'utf8');

    expect(source).not.toMatch(nativeControlPattern);
  });
});
