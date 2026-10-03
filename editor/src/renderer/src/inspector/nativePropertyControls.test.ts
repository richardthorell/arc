import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const inspectorDirectory = dirname(fileURLToPath(import.meta.url));

const migratedPropertySurfaces = [
  'InspectorComponentCard.tsx',
  'InspectorControls.tsx',
  'MaterialParameterSubsection.tsx',
] as const;

const nativePropertyControl = /<(?:input|select|textarea)\b/;

describe('inspector property controls', () => {
  it.each(migratedPropertySurfaces)('%s uses shared ARC UI primitives', (fileName) => {
    const source = readFileSync(resolve(inspectorDirectory, fileName), 'utf8');

    expect(source).not.toMatch(nativePropertyControl);
  });
});
