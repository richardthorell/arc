export type SettingsDialogKind = 'editorPreferences' | 'projectSettings';

export type SettingsDialogOpenRequest = {
  kind: SettingsDialogKind;
  pageId: string | null;
};

const settingsDialogOpenRequestEvent = 'arc:settings-dialog-open-request';

let requestedSettingsDialog: SettingsDialogKind = 'editorPreferences';
let requestedSettingsPageId: string | null = null;

export const requestSettingsDialog = (kind: SettingsDialogKind, pageId: string | null = null) => {
  requestedSettingsDialog = kind;
  requestedSettingsPageId = kind === 'editorPreferences' ? pageId : null;
};

export const requestSettingsDialogOpen = (kind: SettingsDialogKind, pageId: string | null = null) => {
  requestSettingsDialog(kind, pageId);
  window.dispatchEvent(
    new CustomEvent<SettingsDialogOpenRequest>(settingsDialogOpenRequestEvent, {
      detail: { kind, pageId: kind === 'editorPreferences' ? pageId : null },
    }),
  );
};

export const subscribeSettingsDialogOpenRequests = (listener: (request: SettingsDialogOpenRequest) => void) => {
  const onRequest = (event: Event) => listener((event as CustomEvent<SettingsDialogOpenRequest>).detail);
  window.addEventListener(settingsDialogOpenRequestEvent, onRequest);
  return () => window.removeEventListener(settingsDialogOpenRequestEvent, onRequest);
};

export const requestedSettingsDialogKind = () => requestedSettingsDialog;

export const requestedSettingsDialogPageId = () => requestedSettingsPageId;

export const resetSettingsDialogRequest = () => {
  requestedSettingsDialog = 'editorPreferences';
  requestedSettingsPageId = null;
};
