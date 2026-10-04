import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const nativeControlPattern = /<(?:button|input|select|textarea)\b/i;
const nativeControlSelectorPattern = /(?:^|[\s,>+~])(?:button|input|select|textarea)(?=[\s.#:[>+~,{]|$)/im;

describe('material texture sample shared control migration', () => {
  it('keeps the texture sample editor on shared controls', () => {
    const source = readFileSync(new URL('MaterialTextureSampleEditor.tsx', import.meta.url), 'utf8');

    expect(source).toContain('UiToggleButton');
    expect(source).toContain('UiTextInput');
    expect(source).not.toMatch(nativeControlPattern);
  });

  it('keeps texture-sample CSS from restyling native controls', () => {
    const css = readFileSync(new URL('materialTextureSample.css', import.meta.url), 'utf8');

    expect(css).not.toMatch(nativeControlSelectorPattern);
  });
});
