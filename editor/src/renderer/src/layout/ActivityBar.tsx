import { useEffect, useState } from 'react';
import type { ComponentProps, ReactNode } from 'react';
import { Settings } from 'lucide-react';

import { activityRegistry } from '../app/panelRegistry';
import type { ActivityId, ActivityRegistration } from '../app/workbenchTypes';
import { UiButton } from '../ui';

import './ActivityBar.css';

type ActivityBarButtonProps = Omit<ComponentProps<typeof UiButton>, 'children'> & {
  children: ReactNode;
  counter?: number;
  counterLabel?: string;
};

export function ActivityBarButton({
  children,
  className = '',
  counter = 0,
  counterLabel,
  ...props
}: ActivityBarButtonProps) {
  const showCounter = Number.isFinite(counter) && counter > 0;

  return (
    <UiButton className={['activity-button', className].filter(Boolean).join(' ')} {...props}>
      {children}
      {showCounter && (
        <span aria-label={counterLabel ?? `${counter} unread`} className="activity-button-counter">
          {counter}
        </span>
      )}
    </UiButton>
  );
}

type ActivityBarProps = {
  activeActivity: ActivityId;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  onSelectActivity: (activity: ActivityId) => void;
  onSettings: () => void;
};

const utilityActivities = activityRegistry.filter((activity) => activity.id !== 'scene');

export function ActivityBar({
  activeActivity,
  expanded = false,
  onExpandedChange,
  onSelectActivity,
  onSettings,
}: ActivityBarProps) {
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

  const renderActivity = (activity: ActivityRegistration) => {
    const Icon = activity.icon;
    const active = expanded && activeActivity === activity.id;
    const counter = activity.id === 'versionControl' ? versionControlCount : undefined;

    return (
      <ActivityBarButton
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
      </ActivityBarButton>
    );
  };

  return (
    <aside className="activity-bar utility-rail" aria-label="Global utilities">
      <div className="activity-items">{utilityActivities.map(renderActivity)}</div>
      <div className="activity-footer">
        <ActivityBarButton
          aria-label="Settings"
          aria-haspopup="dialog"
          onClick={onSettings}
          title="Settings"
          variant="ghost"
        >
          <Settings size={20} />
        </ActivityBarButton>
      </div>
    </aside>
  );
}
