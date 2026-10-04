import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const nativeFormControlPattern = /<(?:input|select|textarea)\b/i;
const nativeFormControlSelectorPattern = /(?:^|[\s,>+~])(?:input|select|textarea)(?=[\s.#:[>+~,{]|$)/im;

describe('remote asset browser shared control migration', () => {
  it('keeps search, filters, variants, and actions on shared controls', () => {
    const source = readFileSync(new URL('RemoteAssetBrowser.tsx', import.meta.url), 'utf8');

    expect(source).toContain('UiTextInput');
    expect(source).toContain('UiSelect');
    expect(source).toContain('UiIconButton');
    expect(source).toContain('UiButton');
    expect(source).not.toMatch(nativeFormControlPattern);
  });

  it('keeps remote asset CSS from restyling native form controls', () => {
    const css = readFileSync(new URL('remoteAssetBrowser.css', import.meta.url), 'utf8');

    expect(css).not.toMatch(nativeFormControlSelectorPattern);
  });
});
