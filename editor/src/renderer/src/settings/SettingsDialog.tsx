import { useEffect, useState } from 'react';

import { EditorPreferencesDialog } from './EditorPreferencesDialog';
import { ProjectSettingsDialog } from './ProjectSettingsDialog';
import {
  requestedSettingsDialogKind,
  requestedSettingsDialogPageId,
  resetSettingsDialogRequest,
} from './settingsDialogRoute';

type SettingsDialogProps = {
  onClose: () => void;
  onResetLayout: () => void;
};

// Compatibility host for the workbench's existing settings modal slot. Menu
// and utility-rail actions choose which specialized settings dialog should be
// presented before the workbench opens this host.
export function SettingsDialog({ onClose, onResetLayout }: SettingsDialogProps) {
  const [kind] = useState(requestedSettingsDialogKind);
  const [pageId] = useState(requestedSettingsDialogPageId);

  useEffect(() => resetSettingsDialogRequest, []);

  useEffect(() => {
    if (kind !== 'editorPreferences' || !pageId) return;
    const timeout = window.setTimeout(() => {
      window.dispatchEvent(new CustomEvent('arc-settings-navigate', { detail: { id: pageId } }));
    }, 0);
    return () => window.clearTimeout(timeout);
  }, [kind, pageId]);

  const close = () => {
    if (kind === 'editorPreferences') window.dispatchEvent(new Event('arc-editor-settings-closed'));
    resetSettingsDialogRequest();
    onClose();
  };

  if (kind === 'projectSettings') return <ProjectSettingsDialog onClose={close} />;

  return <EditorPreferencesDialog onClose={close} onResetLayout={onResetLayout} />;
}
