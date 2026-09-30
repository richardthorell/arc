// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('./EditorPreferencesDialog', () => ({
  EditorPreferencesDialog: () => <div>Editor preferences</div>,
}));
vi.mock('./ProjectSettingsDialog', () => ({
  ProjectSettingsDialog: () => <div>Project settings</div>,
}));

import { SettingsDialog } from './SettingsDialog';
import { requestSettingsDialog, resetSettingsDialogRequest } from './settingsDialogRoute';

afterEach(() => {
  cleanup();
  resetSettingsDialogRequest();
  vi.useRealTimers();
});

describe('SettingsDialog routing', () => {
  it('navigates to the requested editor-preferences page after the dialog mounts', () => {
    vi.useFakeTimers();
    const navigate = vi.fn();
    window.addEventListener('arc-settings-navigate', navigate);
    requestSettingsDialog('editorPreferences', 'ai.providers');

    render(<SettingsDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    vi.runAllTimers();

    expect(navigate).toHaveBeenCalledOnce();
    expect((navigate.mock.calls[0][0] as CustomEvent).detail).toEqual({ id: 'ai.providers' });
    window.removeEventListener('arc-settings-navigate', navigate);
  });
});
