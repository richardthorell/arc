import { useEffect, useMemo, useState } from 'react';

import { UiTreeView } from '../ui';
import type { UiTreeNode } from '../ui';
import type { InspectorCommand } from './InspectorPanel';
import type { InspectorSkeleton } from './inspectorTypes';

import './skeletonInspector.css';

type SkeletonInspectorProps = {
  skeleton: InspectorSkeleton;
  command: InspectorCommand;
  viewportId?: string;
  onStatus?: (message: string) => void;
};

const formatVector = (values: readonly number[]) => values.map((value) => Number(value.toFixed(4))).join(', ');

function buildTree(skeleton: InspectorSkeleton): UiTreeNode[] {
  const children = new Map<number, InspectorSkeleton['joints']>();
  for (const joint of skeleton.joints) {
    const list = children.get(joint.parent) ?? [];
    list.push(joint);
    children.set(joint.parent, list);
  }
  const build = (joint: InspectorSkeleton['joints'][number]): UiTreeNode => ({
    id: String(joint.index),
    label: joint.name,
    keywords: [joint.name],
    children: (children.get(joint.index) ?? []).map(build),
  });
  return (
    children.get(-1) ??
    skeleton.joints.filter((joint) => !skeleton.joints.some((candidate) => candidate.index === joint.parent))
  ).map(build);
}

export function SkeletonInspector({ skeleton, command, viewportId = 'viewport-1', onStatus }: SkeletonInspectorProps) {
  const fallback = skeleton.joints[0]?.index ?? -1;
  const initial = skeleton.joints.some((joint) => joint.index === skeleton.selectedJoint)
    ? skeleton.selectedJoint
    : fallback;
  const [selectedJoint, setSelectedJoint] = useState(initial);
  useEffect(() => setSelectedJoint(initial), [initial, skeleton.name]);
  const nodes = useMemo(() => buildTree(skeleton), [skeleton]);
  const joint = skeleton.joints.find((candidate) => candidate.index === selectedJoint) ?? skeleton.joints[0];
  const parent = joint?.parent >= 0 ? skeleton.joints.find((candidate) => candidate.index === joint.parent) : undefined;
  const expanded = useMemo(
    () =>
      new Set(
        skeleton.joints.filter((candidate) => candidate.parent >= 0).map((candidate) => String(candidate.parent)),
      ),
    [skeleton],
  );
  if (!joint) return null;

  return (
    <section className="inspector-component-card skeleton-inspector" aria-label="Skeleton">
      <header>
        <strong>Skeleton</strong>
        <span>{skeleton.name}</span>
      </header>
      <UiTreeView
        ariaLabel="Skeleton hierarchy"
        nodes={nodes}
        defaultExpandedIds={[...expanded]}
        selectedId={String(selectedJoint)}
        onSelect={(node) => {
          const next = Number(node.id);
          setSelectedJoint(next);
          void command('viewport.setSkeletonJoint', { viewportId, jointIndex: next }).then((response) => {
            if (!response.succeeded) onStatus?.(response.error || 'Could not select skeleton bone');
            else
              onStatus?.(
                `Selected bone ${skeleton.joints.find((candidate) => candidate.index === next)?.name ?? next}`,
              );
          });
        }}
      />
      <div className="skeleton-inspector-details">
        <label>
          <span>Selected Bone</span>
          <output>{joint.name}</output>
        </label>
        <label>
          <span>Parent</span>
          <output>{parent?.name ?? 'None'}</output>
        </label>
        <fieldset>
          <legend>Bind Transform</legend>
          <label>
            <span>Position</span>
            <output>{formatVector(joint.bindPosition)}</output>
          </label>
          <label>
            <span>Rotation</span>
            <output>{formatVector(joint.bindRotation)}</output>
          </label>
          <label>
            <span>Scale</span>
            <output>{formatVector(joint.bindScale)}</output>
          </label>
        </fieldset>
      </div>
    </section>
  );
}
