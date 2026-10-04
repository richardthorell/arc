import {
  Box,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  FileDiff,
  ListChecks,
  Monitor,
  ShieldCheck,
  Wrench,
} from 'lucide-react';
import { useEffect, useId, useState, type ReactNode } from 'react';

import { UiAgentCard, type UiAgentCardProps, type UiAgentCardState } from './UiAgentCard';
import { UiIconButton } from './UiIconButton';
import './UiAgentActivityCard.css';

export type UiAgentActivityState = 'pending' | 'running' | 'complete' | 'error' | 'cancelled';
export type UiAgentActivityKind = 'task' | 'tool' | 'approval' | 'diff' | 'viewport' | 'asset' | 'error';

export type UiAgentActivityCardProps = Omit<
  UiAgentCardProps,
  'actions' | 'children' | 'icon' | 'side' | 'state' | 'subtitle' | 'title' | 'tone'
> & {
  title: ReactNode;
  subtitle?: ReactNode;
  summary?: ReactNode;
  metadata?: ReactNode;
  details?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  state?: UiAgentActivityState;
  defaultExpanded?: boolean;
  collapsible?: boolean;
};

type ActivityDescriptor = Readonly<{
  label: string;
  icon: ReactNode;
}>;

const activityDescriptors: Record<UiAgentActivityKind, ActivityDescriptor> = {
  task: { label: 'Task', icon: <ListChecks aria-hidden="true" size={15} /> },
  tool: { label: 'Tool', icon: <Wrench aria-hidden="true" size={15} /> },
  approval: { label: 'Approval', icon: <ShieldCheck aria-hidden="true" size={15} /> },
  diff: { label: 'Changes', icon: <FileDiff aria-hidden="true" size={15} /> },
  viewport: { label: 'Viewport', icon: <Monitor aria-hidden="true" size={15} /> },
  asset: { label: 'Asset', icon: <Box aria-hidden="true" size={15} /> },
  error: { label: 'Error', icon: <CircleAlert aria-hidden="true" size={15} /> },
};

const activityStateLabel: Record<UiAgentActivityState, string> = {
  pending: 'Pending',
  running: 'Running',
  complete: 'Complete',
  error: 'Error',
  cancelled: 'Cancelled',
};

const shellStateFor = (state: UiAgentActivityState): UiAgentCardState => {
  if (state === 'error') return 'error';
  if (state === 'pending' || state === 'running') return 'streaming';
  return 'complete';
};

export type UiAgentStructuredCardProps = UiAgentActivityCardProps & {
  kind: UiAgentActivityKind;
};

export function UiAgentStructuredCard({
  kind,
  title,
  subtitle,
  summary,
  metadata,
  details,
  actions,
  children,
  state = 'complete',
  defaultExpanded = false,
  collapsible = true,
  className,
  ...props
}: UiAgentStructuredCardProps) {
  const descriptor = activityDescriptors[kind];
  const detailsId = useId();
  const [expanded, setExpanded] = useState(defaultExpanded);
  const canCollapse = collapsible && Boolean(details);

  useEffect(() => {
    if (defaultExpanded) setExpanded(true);
  }, [defaultExpanded]);

  return (
    <UiAgentCard
      {...props}
      className={['ui-agent-activity-card', `ui-agent-activity-card-${kind}`, className].filter(Boolean).join(' ')}
      data-activity-kind={kind}
      data-activity-state={state}
      icon={descriptor.icon}
      side="none"
      state={shellStateFor(state)}
      subtitle={subtitle ?? descriptor.label}
      title={title}
      tone="neutral"
      actions={
        <>
          <span className="ui-agent-activity-status" data-state={state}>
            {activityStateLabel[state]}
          </span>
          {canCollapse && (
            <UiIconButton
              aria-controls={detailsId}
              aria-expanded={expanded}
              className="ui-agent-activity-disclosure"
              label={expanded ? 'Collapse details' : 'Expand details'}
              type="button"
              variant="ghost"
              onClick={() => setExpanded((current) => !current)}
            >
              {expanded ? <ChevronDown aria-hidden="true" size={13} /> : <ChevronRight aria-hidden="true" size={13} />}
            </UiIconButton>
          )}
          {actions}
        </>
      }
    >
      {(summary || children) && <div className="ui-agent-activity-summary">{summary ?? children}</div>}
      {metadata && <div className="ui-agent-activity-metadata">{metadata}</div>}
      {details && (!canCollapse || expanded) && (
        <div className="ui-agent-activity-details" id={detailsId}>
          {details}
        </div>
      )}
    </UiAgentCard>
  );
}

type FixedActivityCardProps = UiAgentActivityCardProps;

export function UiAgentTaskCard(props: FixedActivityCardProps) {
  return <UiAgentStructuredCard kind="task" {...props} />;
}

export function UiAgentToolCard(props: FixedActivityCardProps) {
  return <UiAgentStructuredCard kind="tool" {...props} />;
}

export function UiAgentApprovalCard(props: FixedActivityCardProps) {
  return <UiAgentStructuredCard kind="approval" {...props} />;
}

export function UiAgentDiffCard(props: FixedActivityCardProps) {
  return <UiAgentStructuredCard kind="diff" {...props} />;
}

export function UiAgentViewportCard(props: FixedActivityCardProps) {
  return <UiAgentStructuredCard kind="viewport" {...props} />;
}

export function UiAgentAssetCard(props: FixedActivityCardProps) {
  return <UiAgentStructuredCard kind="asset" {...props} />;
}

export function UiAgentErrorCard({ state = 'error', ...props }: FixedActivityCardProps) {
  return <UiAgentStructuredCard kind="error" state={state} {...props} />;
}
