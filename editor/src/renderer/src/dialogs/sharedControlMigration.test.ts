import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

const dialogsDirectory = fileURLToPath(new URL('.', import.meta.url));
const dialogSurfaces = readdirSync(dialogsDirectory)
  .filter((filename) => filename.endsWith('Dialog.tsx'))
  .sort();

const nativeControlPattern = /<(?:button|input|select|textarea)\b/i;

describe('dialog shared control migration', () => {
  it('keeps the dialog surface inventory covered', () => {
    expect(dialogSurfaces.length).toBeGreaterThan(0);
  });

  it.each(dialogSurfaces)('%s stays on shared Ui controls', (filename) => {
    const source = readFileSync(new URL(filename, import.meta.url), 'utf8');

    expect(source).not.toMatch(nativeControlPattern);
  });
});
