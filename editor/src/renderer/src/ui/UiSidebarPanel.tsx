import { useEffect, useState } from 'react';
import type { ComponentProps, ReactNode } from 'react';
import { Settings } from 'lucide-react';

import { activityRegistry } from '../app/panelRegistry';
import type { ActivityId, ActivityRegistration } from '../app/workbenchTypes';
import { requestSearchDrawer } from '../search/searchDrawerRoute';
import { requestSettingsDialog } from '../settings/settingsDialogRoute';
import { UiButton } from './UiButton';

import './UiSidebarPanel.css';

export type UiSidebarPanelButtonProps = Omit<ComponentProps<typeof UiButton>, 'children'> & {
  children: ReactNode;
  counter?: number;
  counterLabel?: string;
};

export function UiSidebarPanelButton({
  children,
  className = '',
  counter = 0,
  counterLabel,
  ...props
}: UiSidebarPanelButtonProps) {
  const showCounter = Number.isFinite(counter) && counter > 0;

  return (
    <UiButton
      className={['ui-sidebar-panel-button', 'activity-button', className].filter(Boolean).join(' ')}
      {...props}
    >
      {children}
      {showCounter && (
        <span
          aria-label={counterLabel ?? `${counter} unread`}
          className="ui-sidebar-panel-counter activity-button-counter"
        >
          {counter}
        </span>
      )}
    </UiButton>
  );
}

export type UiSidebarPanelProps = {
  activeActivity: ActivityId;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  onSelectActivity: (activity: ActivityId) => void;
  onSettings: () => void;
};

const utilityActivities = activityRegistry.filter((activity) => activity.id !== 'scene');

/** Fixed utility rail that opens document-independent UiDrawerPanel surfaces. */
export function UiSidebarPanel({
  activeActivity,
  expanded = false,
  onExpandedChange,
  onSelectActivity,
  onSettings,
}: UiSidebarPanelProps) {
  const [versionControlCount, setVersionControlCount] = useState(0);

  useEffect(() => {
    let disposed = false;

    const refreshVersionControlCount = () => {
      const snapshot = window.arc?.sourceControl?.snapshot;
      if (!snapshot) return;

      void snapshot()
        .then((next) => {
          if (!disposed) setVersionControlCount(next?.available ? next.files.length : 0);
        })
        .catch(() => {
          if (!disposed) setVersionControlCount(0);
        });
    };

    refreshVersionControlCount();
    const interval = window.setInterval(refreshVersionControlCount, 2500);
    window.addEventListener('focus', refreshVersionControlCount);

    return () => {
      disposed = true;
      window.clearInterval(interval);
      window.removeEventListener('focus', refreshVersionControlCount);
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.repeat ||
        !event.ctrlKey ||
        event.altKey ||
        event.metaKey ||
        event.key.toLocaleLowerCase() !== 'p'
      ) {
        return;
      }

      event.preventDefault();
      event.stopImmediatePropagation();
      const mode = event.shiftKey ? 'commands' : 'assets';
      onSelectActivity('search');
      onExpandedChange?.(true);
      window.setTimeout(() => requestSearchDrawer(mode), 0);
    };

    window.addEventListener('keydown', onKeyDown, true);
    return () => window.removeEventListener('keydown', onKeyDown, true);
  }, [onExpandedChange, onSelectActivity]);

  const renderActivity = (activity: ActivityRegistration) => {
    const Icon = activity.icon;
    const active = expanded && activeActivity === activity.id;
    const counter = activity.id === 'versionControl' ? versionControlCount : undefined;

    return (
      <UiSidebarPanelButton
        active={active}
        aria-label={activity.title}
        aria-pressed={active}
        counter={counter}
        counterLabel={counter ? `${counter} changed ${counter === 1 ? 'file' : 'files'}` : undefined}
        key={activity.id}
        onClick={() => {
          if (activeActivity === activity.id && expanded) {
            onExpandedChange?.(false);
            return;
          }
          onSelectActivity(activity.id);
          onExpandedChange?.(true);
        }}
        title={activity.title}
        variant="ghost"
      >
        <Icon size={20} />
      </UiSidebarPanelButton>
    );
  };

  return (
    <aside className="ui-sidebar-panel activity-bar utility-rail" aria-label="Global utilities">
      <div className="ui-sidebar-panel-items activity-items">{utilityActivities.map(renderActivity)}</div>
      <div className="ui-sidebar-panel-footer activity-footer">
        <UiSidebarPanelButton
          aria-label="Editor Preferences"
          aria-haspopup="dialog"
          onClick={() => {
            requestSettingsDialog('editorPreferences');
            onSettings();
          }}
          title="Editor Preferences"
          variant="ghost"
        >
          <Settings size={20} />
        </UiSidebarPanelButton>
      </div>
    </aside>
  );
}
