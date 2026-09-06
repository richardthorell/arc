import { useMemo, useState } from 'react';
import { Bone, Box, CheckCircle2, ExternalLink, TriangleAlert, XCircle } from 'lucide-react';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from '../assetPreview/AssetPreviewPanel';
import { openSkeletonEditorDocument } from '../editors/editorRegistry';
import type { EditorDocument } from '../editors/editorTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { UiButton, UiPanel } from '../ui';
import { buildModelSubassets, hasSkeletonMetadata, skeletonCompatibility } from './modelSubassets';

import './modelEditor.css';

type ProjectAssetsResponse = { succeeded?: boolean; payload?: { assets?: AssetItem[] } };

const compatibilityIcon = (value: ReturnType<typeof skeletonCompatibility>) =>
  value === 'compatible' ? (
    <CheckCircle2 size={13} />
  ) : value === 'partial' ? (
    <TriangleAlert size={13} />
  ) : (
    <XCircle size={13} />
  );

export function ModelEditor({ document }: { document: EditorDocument }) {
  const asset = document.assetSnapshot;
  const [activeSection, setActiveSection] = useState<'model' | 'skeleton'>('model');
  const [candidateAssets, setCandidateAssets] = useState<AssetItem[]>([]);
  const [assignedSkeleton, setAssignedSkeleton] = useState(asset?.guid ?? asset?.path ?? '');
  const subassets = useMemo(() => (asset ? buildModelSubassets(asset) : []), [asset]);

  if (!asset) return <div className="model-editor-empty">Model metadata is unavailable.</div>;
  const skeleton = hasSkeletonMetadata(asset);
  const boneCount = asset.skeletonBoneCount ?? asset.skeletonJoints?.length ?? 0;

  const loadSkeletonCandidates = async () => {
    try {
      const response = (await window.arc.host.query('project.assets')) as ProjectAssetsResponse;
      setCandidateAssets((response.payload?.assets ?? []).filter(hasSkeletonMetadata));
    } catch {
      setCandidateAssets([]);
    }
  };

  return (
    <div className="model-editor-shell">
      <AssetPreviewPanel
        title={asset.name}
        subtitle={asset.path}
        metadata={[
          { label: 'Meshes', value: asset.meshCount ?? 0 },
          { label: 'Vertices', value: asset.vertexCount?.toLocaleString() ?? '—' },
          { label: 'Triangles', value: asset.triangleCount?.toLocaleString() ?? '—' },
          { label: 'Skeleton', value: skeleton ? `${boneCount} bones` : 'None' },
        ]}
      >
        <AssetPreviewPlaceholder
          label="Model preview"
          description={skeleton ? 'Skinned model with imported skeleton metadata.' : 'Static model preview.'}
        />
      </AssetPreviewPanel>

      <UiPanel
        aria-label="Model details"
        className="model-editor-sidebar editor-property-panel"
        role="complementary"
        variant="inspector"
      >
        <div className="model-editor-tabs" role="tablist" aria-label="Model editor sections">
          <button className={activeSection === 'model' ? 'active' : ''} onClick={() => setActiveSection('model')}>
            <Box size={14} /> Model
          </button>
          <button
            className={activeSection === 'skeleton' ? 'active' : ''}
            disabled={!skeleton}
            onClick={() => setActiveSection('skeleton')}
          >
            <Bone size={14} /> Skeleton
          </button>
        </div>

        {activeSection === 'model' ? (
          <div className="model-editor-section">
            <h3>Sub-assets</h3>
            <div className="model-subasset-list">
              {subassets.map((subasset) => (
                <button
                  key={subasset.id}
                  className="model-subasset-row"
                  onDoubleClick={() => subasset.kind === 'skeleton' && openSkeletonEditorDocument(asset)}
                >
                  {subasset.kind === 'skeleton' ? <Bone size={14} /> : <Box size={14} />}
                  <span>{subasset.name}</span>
                  <small>{subasset.kind === 'skeleton' ? 'Skeleton' : 'Mesh'}</small>
                </button>
              ))}
              {!subassets.length && <p>No model sub-assets were reported by the importer.</p>}
            </div>
          </div>
        ) : (
          <div className="model-editor-section">
            <div className="model-editor-section-heading">
              <h3>{asset.skeletonName || 'Skeleton'}</h3>
              <UiButton type="button" variant="toolbar" onClick={() => openSkeletonEditorDocument(asset)}>
                Open <ExternalLink size={12} />
              </UiButton>
            </div>
            <dl className="model-skeleton-summary">
              <div>
                <dt>Bones</dt>
                <dd>{boneCount}</dd>
              </div>
              <div>
                <dt>Root</dt>
                <dd>
                  {asset.skeletonRootBone || asset.skeletonJoints?.find((joint) => joint.parent < 0)?.name || '—'}
                </dd>
              </div>
              <div>
                <dt>Hierarchy depth</dt>
                <dd>{asset.skeletonHierarchyDepth ?? '—'}</dd>
              </div>
              <div>
                <dt>Bind pose</dt>
                <dd>{asset.skeletonJoints?.length ? 'Available' : 'Metadata only'}</dd>
              </div>
            </dl>
            <label className="model-skeleton-picker">
              <span>Skeleton assignment</span>
              <select
                value={assignedSkeleton}
                onFocus={() => void loadSkeletonCandidates()}
                onChange={(event) => setAssignedSkeleton(event.target.value)}
              >
                <option value={asset.guid ?? asset.path}>
                  {asset.skeletonName || `${asset.name} Skeleton`} (Imported)
                </option>
                {candidateAssets
                  .filter((candidate) => (candidate.guid ?? candidate.path) !== (asset.guid ?? asset.path))
                  .map((candidate) => {
                    const compatibility = skeletonCompatibility(asset, candidate);
                    return (
                      <option
                        key={candidate.guid ?? candidate.path}
                        value={candidate.guid ?? candidate.path}
                        disabled={compatibility === 'incompatible'}
                      >
                        {candidate.skeletonName || candidate.name} · {compatibility}
                      </option>
                    );
                  })}
              </select>
            </label>
            <div className="model-compatibility-legend">
              <span>{compatibilityIcon('compatible')} compatible</span>
              <span>{compatibilityIcon('partial')} partially compatible</span>
              <span>{compatibilityIcon('incompatible')} incompatible</span>
            </div>
            <p className="model-editor-note">Full animation retargeting is intentionally outside this milestone.</p>
          </div>
        )}
      </UiPanel>
    </div>
  );
}

export function ModelEditorToolbar() {
  return <span className="model-editor-toolbar-label">Model</span>;
}
