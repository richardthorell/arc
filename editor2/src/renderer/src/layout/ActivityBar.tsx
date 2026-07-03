import { activityRegistry } from '../app/panelRegistry';
import type { ActivityId } from '../app/workbenchTypes';

type ActivityBarProps = {
  activeActivity: ActivityId;
  onSelectActivity: (activity: ActivityId) => void;
};

export function ActivityBar({ activeActivity, onSelectActivity }: ActivityBarProps) {
  return (
    <aside className="activity-bar" aria-label="Editor activity bar">
      {activityRegistry.map((activity, index) => {
        const Icon = activity.icon;
        const spacerBefore = index === 5;
        return (
          <div className={spacerBefore ? 'activity-group-spaced' : undefined} key={activity.id}>
            <button
              className={activeActivity === activity.id ? 'activity-button active' : 'activity-button'}
              title={activity.title}
              onClick={() => onSelectActivity(activity.id)}
            >
              <Icon size={22} />
            </button>
          </div>
        );
      })}
    </aside>
  );
}
