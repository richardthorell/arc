import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const nativeFormControlPattern = /<(?:input|select|textarea)\b/i;
const nativeMainControlSelectorPattern =
  /(?:\.content-browser-search|\.content-toolbar(?=[\s.#:[>+~,{]))[^{]*\b(?:input|select|textarea)\b/im;

describe('content browser shared control migration', () => {
  it('keeps search, filters, sorting, and actions on shared controls', () => {
    const source = readFileSync(new URL('ContentBrowserPanelCore.tsx', import.meta.url), 'utf8');
    const createDialogStart = source.indexOf('{createKind && (');

    expect(createDialogStart).toBeGreaterThan(0);
    const browserSurface = source.slice(0, createDialogStart);

    expect(browserSurface).toContain('UiSearchInput');
    expect(browserSurface).toContain('UiSelect');
    expect(browserSurface).toContain('UiIconButton');
    expect(browserSurface).toContain('UiButton');
    expect(browserSurface).not.toMatch(nativeFormControlPattern);
  });

  it('keeps main content browser CSS from restyling native form controls', () => {
    const css = readFileSync(new URL('contentBrowser.css', import.meta.url), 'utf8');

    expect(css).not.toMatch(nativeMainControlSelectorPattern);
  });
});
