import { useEffect, useMemo, useState } from 'react';

import type { AssetItem } from '../services/editorHostTypes';
import { UiButton, UiDialog, UiTextArea, UiTextInput } from '../ui';
import { assetDisplayName, assetSpecificHoverDetails } from './ContentAssetCard';
import { assetPresentationLabel } from './assetPresentation';
import { normalizeAssetTags, type AssetMetadata } from './assetMetadataStore';

import './assetMetadataDialog.css';

type Props = {
  asset: AssetItem;
  onClose: () => void;
  onSave: (metadata: AssetMetadata) => Promise<void>;
};

export function AssetMetadataDialog({ asset, onClose, onSave }: Props) {
  const [title, setTitle] = useState(asset.title ?? '');
  const [description, setDescription] = useState(asset.description ?? '');
  const [tags, setTags] = useState((asset.tags ?? []).join(', '));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const details = useMemo(() => assetSpecificHoverDetails(asset), [asset]);
  const editable = !asset.readOnly && (asset.scope ?? 'project') === 'project';

  useEffect(() => {
    setTitle(asset.title ?? '');
    setDescription(asset.description ?? '');
    setTags((asset.tags ?? []).join(', '));
    setError('');
  }, [asset]);

  const submit = async () => {
    if (!editable || saving) return;
    setSaving(true);
    setError('');
    try {
      await onSave({
        title,
        description,
        tags: normalizeAssetTags(tags.split(',')),
      });
      onClose();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setSaving(false);
    }
  };

  return (
    <UiDialog
      ariaLabel="Asset metadata"
      className="asset-metadata-dialog"
      footer={
        <>
          <UiButton type="button" variant="ghost" onClick={onClose}>
            {editable ? 'Cancel' : 'Close'}
          </UiButton>
          {editable && (
            <UiButton type="button" disabled={saving} onClick={() => void submit()}>
              {saving ? 'Saving…' : 'Save metadata'}
            </UiButton>
          )}
        </>
      }
      onClose={onClose}
      subtitle={`${assetPresentationLabel(asset)} · ${asset.path}`}
      title={assetDisplayName(asset)}
      width={520}
    >
      <div className="asset-metadata-dialog-content">
        <section className="asset-metadata-section" aria-label="Search metadata">
          <header>
            <strong>Search metadata</strong>
            <small>Stored with the project and independent of Content Browser layout.</small>
          </header>
          <label>
            <span>Title</span>
            <UiTextInput
              aria-label="Asset title"
              disabled={!editable || saving}
              placeholder={asset.name}
              value={title}
              onChange={(event) => setTitle(event.target.value)}
            />
          </label>
          <label>
            <span>Description</span>
            <UiTextArea
              aria-label="Asset description"
              disabled={!editable || saving}
              rows={3}
              value={description}
              onChange={(event) => setDescription(event.target.value)}
            />
          </label>
          <label>
            <span>Tags</span>
            <UiTextInput
              aria-label="Asset tags"
              disabled={!editable || saving}
              placeholder="environment, hero, gameplay"
              value={tags}
              onChange={(event) => setTags(event.target.value)}
            />
            <small>Comma-separated. Tags are normalized and de-duplicated when saved.</small>
          </label>
          {!editable && (
            <p className="asset-metadata-readonly">Built-in and read-only asset metadata cannot be edited.</p>
          )}
          {error && <p className="asset-metadata-error">{error}</p>}
        </section>

        <section className="asset-metadata-section" aria-label={`${assetPresentationLabel(asset)} metadata`}>
          <header>
            <strong>{assetPresentationLabel(asset)} details</strong>
            <small>Type-specific metadata reported by the asset registry.</small>
          </header>
          <dl className="asset-metadata-details">
            <div>
              <dt>Status</dt>
              <dd>{asset.status}</dd>
            </div>
            {asset.typeId && (
              <div>
                <dt>Type ID</dt>
                <dd>{asset.typeId}</dd>
              </div>
            )}
            {asset.importerId && (
              <div>
                <dt>Importer</dt>
                <dd>{asset.importerId}</dd>
              </div>
            )}
            {details.map((detail) => (
              <div key={detail.label}>
                <dt>{detail.label}</dt>
                <dd>{detail.value}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
    </UiDialog>
  );
}
