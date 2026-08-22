import type { ReactNode } from 'react';

import './AssetPreviewPanel.css';

export interface AssetPreviewMetadataItem {
  label: string;
  value: ReactNode;
}

interface AssetPreviewPanelProps {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
  children?: ReactNode;
  metadata?: readonly AssetPreviewMetadataItem[];
}

interface AssetPreviewPlaceholderProps {
  label?: string;
  description?: string;
}

export function AssetPreviewPlaceholder({
  label = 'Preview viewport',
  description = 'Native preview surface integration pending.',
}: AssetPreviewPlaceholderProps) {
  return (
    <div className="asset-preview-placeholder">
      <div className="asset-preview-placeholder-widget" aria-hidden="true">
        <span className="asset-preview-placeholder-orb" />
        <span className="asset-preview-placeholder-ground" />
      </div>
      <strong>{label}</strong>
      <span>{description}</span>
    </div>
  );
}

export function AssetPreviewPanel({ title, subtitle, actions, children, metadata = [] }: AssetPreviewPanelProps) {
  return (
    <section className="asset-preview-panel" aria-label={title}>
      <header className="asset-preview-header">
        <div className="asset-preview-heading">
          <strong>{title}</strong>
          {subtitle && <span>{subtitle}</span>}
        </div>
        {actions && <div className="asset-preview-actions">{actions}</div>}
      </header>
      <div className="asset-preview-stage">{children ?? <AssetPreviewPlaceholder />}</div>
      {metadata.length > 0 && (
        <footer className="asset-preview-metadata">
          {metadata.map((item, index) => (
            <span className="asset-preview-metadata-item" key={`${item.label}-${index}`}>
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </span>
          ))}
        </footer>
      )}
    </section>
  );
}
