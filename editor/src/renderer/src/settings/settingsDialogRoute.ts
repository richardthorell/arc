export type SettingsDialogKind = 'editorPreferences' | 'projectSettings';

let requestedSettingsDialog: SettingsDialogKind = 'editorPreferences';

export const requestSettingsDialog = (kind: SettingsDialogKind) => {
  requestedSettingsDialog = kind;
};

export const requestedSettingsDialogKind = () => requestedSettingsDialog;

export const resetSettingsDialogRequest = () => {
  requestedSettingsDialog = 'editorPreferences';
};
