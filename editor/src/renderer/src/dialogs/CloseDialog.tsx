import { AlertTriangle } from 'lucide-react';

import { UiButton, UiDialog } from '../ui';

export type CloseDialogChoice = 'save' | 'discard' | 'cancel';

type CloseDialogProps = {
  onChoose: (choice: CloseDialogChoice) => void;
  preview?: boolean;
  sceneName?: string;
};

export function CloseDialog({ onChoose, preview = false, sceneName = 'Untitled' }: CloseDialogProps) {
  return (
    <UiDialog
      footer={
        <>
          <UiButton onClick={() => onChoose('cancel')}>Cancel</UiButton>
          <UiButton onClick={() => onChoose('discard')}>Don't Save</UiButton>
          <UiButton onClick={() => onChoose('save')} variant="primary">
            Save
          </UiButton>
        </>
      }
      icon={<AlertTriangle aria-hidden="true" size={18} />}
      onClose={() => onChoose('cancel')}
      preview={preview}
      subtitle="Unsaved changes"
      title="Close ARC Project"
      width={520}
    >
      <div className="close-dialog-copy">
        <strong>Save changes to {sceneName} before closing?</strong>
        <p>Unsaved scene authoring changes will be lost.</p>
      </div>
    </UiDialog>
  );
}
