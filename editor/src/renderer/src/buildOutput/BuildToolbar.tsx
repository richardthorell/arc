import { useEffect, useState } from 'react';
import {
  Hammer,
  RefreshCw,
  Settings2,
  Square,
  Trash2,
  Flame,
  Package,
} from 'lucide-react';

import './BuildToolbar.css';

import type { ArcBuildSnapshot } from '../../../common/buildTypes';
import type { WorkflowExecutionSnapshot } from '../../../common/projectWorkflowTypes';
import { UiButton, UiSplitButton, type UiSplitButtonOption } from '../ui';

export type BuildDialogAction = 'build' | 'rebuild' | 'cook' | 'configure' | 'clean';

const actions: ReadonlyArray<UiSplitButtonOption<BuildDialogAction | 'package'>> = [
  { value: 'build', label: 'Build...', icon: <Hammer size={14} /> },
  { value: 'rebuild', label: 'Rebuild editor code...', icon: <RefreshCw size={14} /> },
  { value: 'cook', label: 'Cook content...', icon: <Flame size={14} /> },
  {
    value: 'package',
    label: 'Package separately (not supported yet)',
    icon: <Package size={14} />,
    disabled: true,
  },
  { value: 'configure', label: 'Configure editor code...', icon: <Settings2 size={14} /> },
  { value: 'clean', label: 'Clean editor code...', icon: <Trash2 size={14} /> },
];

const codeBusy = (snapshot: ArcBuildSnapshot | null) =>
  snapshot?.state === 'building' || snapshot?.state === 'configuring' || snapshot?.state === 'cleaning';

const workflowBusy = (snapshot: WorkflowExecutionSnapshot | null) =>
  Boolean(
    snapshot?.planning ||
      snapshot?.run?.state === 'running' ||
      snapshot?.run?.state === 'cancelling',
  );

/** Builds-only editor toolbar. The actual process lives in Electron, not this component. */
export function BuildToolbar({
  snapshot,
  workflow,
  onCreate,
}: {
  snapshot: ArcBuildSnapshot | null;
  workflow: WorkflowExecutionSnapshot | null;
  onCreate: (action: BuildDialogAction) => void;
}) {
  const [stopping, setStopping] = useState(false);
  const [error, setError] = useState('');
  const busy = codeBusy(snapshot) || workflowBusy(workflow);
  const cancelling = stopping || workflow?.run?.state === 'cancelling';

  useEffect(() => {
    if (!busy) setStopping(false);
  }, [busy]);

  const stop = async () => {
    if (stopping || !busy) return;
    setError('');
    setStopping(true);
    try {
      if (workflowBusy(workflow)) {
        if (!workflow?.requestId) throw new Error('Build has not provided a cancellation ID yet');
        await window.arc.build.cancelWorkflow(workflow.requestId);
      } else {
        await window.arc.build.execute({ action: 'cancel' });
      }
    } catch (reason) {
      setStopping(false);
      setError(String(reason));
    }
  };

  return (
    <div className="builds-editor-toolbar" aria-label="Builds toolbar">
      {busy ? (
        <UiButton
          className="builds-toolbar-stop"
          variant="danger"
          disabled={cancelling}
          onClick={() => void stop()}
          type="button"
        >
          <Square size={13} fill="currentColor" />
          {cancelling ? 'Stopping...' : 'Stop'}
        </UiButton>
      ) : (
        <UiSplitButton
          label="Build..."
          ariaLabel="Create build"
          menuAriaLabel="Build operations"
          icon={<Hammer size={14} />}
          options={actions}
          onClick={() => onCreate('build')}
          onOptionSelect={(action) => {
            if (action !== 'package') onCreate(action);
          }}
        />
      )}
      {busy && (
        <span className="builds-toolbar-progress" role="status">
          {workflow?.planning
            ? 'Planning build'
            : workflowBusy(workflow)
              ? workflow?.run?.stages.find((stage) => stage.state === 'running')?.stage ?? 'Building'
              : snapshot?.state ?? 'Building'}
        </span>
      )}
      <span className="builds-toolbar-spacer" />
      {error && <span className="builds-toolbar-error" role="alert">{error}</span>}
    </div>
  );
}
