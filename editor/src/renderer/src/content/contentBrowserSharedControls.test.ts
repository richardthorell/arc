import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const nativeFormControlPattern = /<(?:input|select|textarea)\b/i;
const nativeFormControlSelectorPattern = /(?:^|[\s,>+~])(?:input|select|textarea)(?=[\s.#:[>+~,{]|$)/im;

describe('content browser shared control migration', () => {
  it('keeps search, filters, sorting, and actions on shared controls', () => {
    const source = readFileSync(new URL('ContentBrowserPanelCore.tsx', import.meta.url), 'utf8');

    expect(source).toContain('UiSearchInput');
    expect(source).toContain('UiSelect');
    expect(source).toContain('UiIconButton');
    expect(source).toContain('UiButton');
    expect(source).not.toMatch(nativeFormControlPattern);
  });

  it('keeps content browser CSS from restyling native form controls', () => {
    const css = readFileSync(new URL('contentBrowser.css', import.meta.url), 'utf8');

    expect(css).not.toMatch(nativeFormControlSelectorPattern);
  });
});
