import { PanelLeftClose, PanelLeftOpen } from 'lucide-react';

import { activityRegistry } from '../app/panelRegistry';
import type { ActivityId } from '../app/workbenchTypes';
import { UiButton, UiIconButton } from '../ui';

type ActivityBarProps = {
  activeActivity: ActivityId;
  expanded: boolean;
  onExpandedChange: (expanded: boolean) => void;
  onSelectActivity: (activity: ActivityId) => void;
};

export function ActivityBar({ activeActivity, expanded, onExpandedChange, onSelectActivity }: ActivityBarProps) {
  return (
    <aside className={expanded ? 'activity-bar activity-bar-expanded' : 'activity-bar'} aria-label="Editor activity bar">
      <div className="activity-items">
        {activityRegistry.map((activity) => {
          const Icon = activity.icon;
          const spacerBefore = activity.id === 'settings';
          return (
            <div className={spacerBefore ? 'activity-group-spaced' : undefined} key={activity.id}>
              <UiButton
                active={activeActivity === activity.id}
                className="activity-button"
                title={activity.title}
                onClick={() => onSelectActivity(activity.id)}
                variant="ghost"
              >
                <Icon size={20} />
                {expanded && <span>{activity.title}</span>}
              </UiButton>
            </div>
          );
        })}
      </div>
      <UiIconButton
        className="activity-collapse-button"
        label={expanded ? 'Collapse activity bar' : 'Expand activity bar'}
        onClick={() => onExpandedChange(!expanded)}
      >
        {expanded ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
      </UiIconButton>
    </aside>
  );
}
