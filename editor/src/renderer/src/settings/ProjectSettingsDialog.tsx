import { UiDialogSettings } from '../ui';

export function ProjectSettingsDialog({ onClose }: { onClose: () => void }) {
  return <UiDialogSettings onClose={onClose} subtitle="Settings shared by this project" title="Project Settings" />;
}
