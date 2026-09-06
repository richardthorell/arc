import { useMemo } from 'react';
import { Bone, TriangleAlert } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { UiTreeView } from '../ui';
import type { UiTreeNode } from '../ui';
import { skeletonHierarchyDepth } from './modelSubassets';

import './modelEditor.css';

function skeletonTree(document: EditorDocument): UiTreeNode[] {
  const joints = document.assetSnapshot?.skeletonJoints ?? [];
  const children = new Map<number, typeof joints>();
  for (const joint of joints) children.set(joint.parent, [...(children.get(joint.parent) ?? []), joint]);
  const build = (joint: (typeof joints)[number]): UiTreeNode => ({
    id: String(joint.index),
    label: joint.name,
    children: (children.get(joint.index) ?? []).map(build),
  });
  return (children.get(-1) ?? []).map(build);
}

export function SkeletonAssetEditor({ document }: { document: EditorDocument }) {
  const asset = document.assetSnapshot;
  const nodes = useMemo(() => skeletonTree(document), [document]);
  if (!asset) return <div className="model-editor-empty">Skeleton metadata is unavailable.</div>;
  const boneCount = asset.skeletonBoneCount ?? asset.skeletonJoints?.length ?? 0;
  const root = asset.skeletonRootBone || asset.skeletonJoints?.find((joint) => joint.parent < 0)?.name || '—';
  const warnings = asset.skeletonValidationWarnings ?? [];

  return (
    <div className="skeleton-asset-editor">
      <header>
        <div>
          <Bone size={18} />
          <div>
            <strong>{asset.skeletonName || `${asset.name} Skeleton`}</strong>
            <span>Model sub-asset · {asset.name}</span>
          </div>
        </div>
      </header>
      <div className="skeleton-asset-body">
        <section className="skeleton-tree-card">
          <h3>Skeleton</h3>
          {nodes.length ? (
            <UiTreeView
              ariaLabel="Skeleton asset hierarchy"
              nodes={nodes}
              defaultExpandedIds={nodes.map((node) => node.id)}
            />
          ) : (
            <p>Joint hierarchy metadata was not emitted by the importer.</p>
          )}
        </section>
        <aside className="skeleton-asset-details">
          <h3>Details</h3>
          <dl>
            <div>
              <dt>Bone count</dt>
              <dd>{boneCount}</dd>
            </div>
            <div>
              <dt>Hierarchy depth</dt>
              <dd>{skeletonHierarchyDepth(asset) || '—'}</dd>
            </div>
            <div>
              <dt>Root bone</dt>
              <dd>{root}</dd>
            </div>
            <div>
              <dt>Bind pose</dt>
              <dd>{asset.skeletonJoints?.length ? 'Available' : 'Metadata only'}</dd>
            </div>
            <div>
              <dt>Model reference</dt>
              <dd>{asset.name}</dd>
            </div>
          </dl>
          <h3>Validation</h3>
          {warnings.length ? (
            warnings.map((warning) => (
              <div className="skeleton-warning" key={warning}>
                <TriangleAlert size={13} />
                {warning}
              </div>
            ))
          ) : (
            <p>No validation warnings.</p>
          )}
        </aside>
      </div>
    </div>
  );
}

export function SkeletonAssetEditorToolbar() {
  return <span className="model-editor-toolbar-label">Skeleton</span>;
}
