import { useEffect, type ReactNode } from 'react';
import { Settings } from 'lucide-react';

import { UiDialog } from './UiDialog';

import './UiDialogSettings.css';

export type UiDialogSettingsProps = {
  children?: ReactNode;
  icon?: ReactNode;
  message?: ReactNode;
  onClose: () => void;
  sidebar?: ReactNode;
  subtitle?: string;
  title: string;
};

export function UiDialogSettings({
  children,
  icon,
  message,
  onClose,
  sidebar,
  subtitle,
  title,
}: UiDialogSettingsProps) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      onClose();
    };

    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [onClose]);

  return (
    <UiDialog
      ariaLabel={title}
      className="ui-dialog-settings"
      draggable={false}
      footer={message ? <div className="ui-dialog-settings-message">{message}</div> : undefined}
      icon={icon ?? <Settings aria-hidden="true" size={18} />}
      onClose={onClose}
      subtitle={subtitle}
      title={title}
      width={1040}
    >
      <div className={`ui-dialog-settings-layout${sidebar ? '' : ' is-single-pane'}`}>
        {sidebar && <aside className="ui-dialog-settings-sidebar">{sidebar}</aside>}
        <div className="ui-dialog-settings-content">{children}</div>
      </div>
    </UiDialog>
  );
}
