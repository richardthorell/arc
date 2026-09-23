import { useEffect, useState } from 'react';

import { EditorPreferencesDialog } from './EditorPreferencesDialog';
import { ProjectSettingsDialog } from './ProjectSettingsDialog';
import { requestedSettingsDialogKind, resetSettingsDialogRequest } from './settingsDialogRoute';

type SettingsDialogProps = {
  onClose: () => void;
  onResetLayout: () => void;
};

// Compatibility host for the workbench's existing settings modal slot. Menu
// and utility-rail actions choose which specialized settings dialog should be
// presented before the workbench opens this host.
export function SettingsDialog({ onClose, onResetLayout }: SettingsDialogProps) {
  const [kind] = useState(requestedSettingsDialogKind);

  useEffect(() => resetSettingsDialogRequest, []);

  const close = () => {
    resetSettingsDialogRequest();
    onClose();
  };

  if (kind === 'projectSettings') return <ProjectSettingsDialog onClose={close} />;

  return <EditorPreferencesDialog onClose={close} onResetLayout={onResetLayout} />;
}
