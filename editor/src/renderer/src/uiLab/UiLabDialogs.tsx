import type { ReactNode } from 'react';

import { CloseDialog } from '../dialogs/CloseDialog';
import { ImportDialog } from '../dialogs/ImportDialog';
import { SettingsDialog } from '../settings/SettingsDialog';
import { UiDialog } from '../ui';

import './uiLabDialogs.css';

function DialogPreview({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="ui-lab-dialog-preview">
      <header>
        <strong>{title}</strong>
      </header>
      <div>{children}</div>
    </section>
  );
}

export function UiLabDialogs() {
  return (
    <main className="ui-lab-dialogs">
      <header className="ui-lab-dialogs-heading">
        <div>
          <strong>Dialog Lab</strong>
          <span>Reusable dialog foundations and production dialog previews.</span>
        </div>
      </header>

      <DialogPreview title="Base dialog">
        <UiDialog preview width={520} />
      </DialogPreview>

      <DialogPreview title="Close dialog">
        <CloseDialog onChoose={() => undefined} preview sceneName="Example Scene" />
      </DialogPreview>

      <DialogPreview title="Import dialog">
        <ImportDialog onClose={() => undefined} onImport={() => undefined} preview />
      </DialogPreview>

      <DialogPreview title="Settings dialog">
        <SettingsDialog onClose={() => undefined} onResetLayout={() => undefined} />
      </DialogPreview>
    </main>
  );
}
