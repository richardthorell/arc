import { activityRegistry } from '../app/panelRegistry';
import type { ActivityId } from '../app/workbenchTypes';
import { UiButton } from '../ui';

type ActivityBarProps = {
  activeActivity: ActivityId;
  /** Kept temporarily for Workbench state compatibility; the rail is now always icon-only. */
  expanded?: boolean;
  /** Kept temporarily for Workbench state compatibility; expansion is no longer exposed. */
  onExpandedChange?: (expanded: boolean) => void;
  onSelectActivity: (activity: ActivityId) => void;
};

export function ActivityBar({ activeActivity, onSelectActivity }: ActivityBarProps) {
  return (
    <aside className="activity-bar" aria-label="Primary sidebar activities">
      <div className="activity-items">
        {activityRegistry.map((activity) => {
          const Icon = activity.icon;
          return (
            <UiButton
              active={activeActivity === activity.id}
              aria-label={activity.title}
              className="activity-button"
              key={activity.id}
              onClick={() => onSelectActivity(activity.id)}
              title={activity.title}
              variant="ghost"
            >
              <Icon size={20} />
            </UiButton>
          );
        })}
      </div>
    </aside>
  );
}
