import { activityRegistry } from '../app/panelRegistry';
import type { ActivityId, ActivityRegistration } from '../app/workbenchTypes';
import { UiButton } from '../ui';

type ActivityBarProps = {
  activeActivity: ActivityId;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  onSelectActivity: (activity: ActivityId) => void;
};

const utilityActivities = activityRegistry.filter((activity) => activity.id !== 'scene');
const primaryActivities = utilityActivities.filter((activity) => activity.id !== 'settings');
const footerActivities = utilityActivities.filter((activity) => activity.id === 'settings');

export function ActivityBar({
  activeActivity,
  expanded = false,
  onExpandedChange,
  onSelectActivity,
}: ActivityBarProps) {
  const renderActivity = (activity: ActivityRegistration) => {
    const Icon = activity.icon;
    const active = expanded && activeActivity === activity.id;
    return (
      <UiButton
        active={active}
        aria-label={activity.title}
        aria-pressed={active}
        className="activity-button"
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
      </UiButton>
    );
  };

  return (
    <aside className="activity-bar utility-rail" aria-label="Global utilities">
      <div className="activity-items">{primaryActivities.map(renderActivity)}</div>
      <div className="activity-footer">{footerActivities.map(renderActivity)}</div>
    </aside>
  );
}
