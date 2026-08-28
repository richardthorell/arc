import {
  UiLabMaterialGraph,
  UiLabMaterialNodeCard,
  UiLabMaterialParameters,
  UiLabMaterialPreview,
} from './UiLabMaterialSurfaces';

import './uiLabMaterialGallery.css';

function ControlCard({ title, caption, children }: { title: string; caption: string; children: React.ReactNode }) {
  return (
    <article className="ui-lab-card ui-lab-card-wide">
      <header>
        <strong>{title}</strong>
        <code>{caption}</code>
      </header>
      <div className="ui-lab-card-stage">{children}</div>
    </article>
  );
}

function SurfaceCard({
  title,
  component,
  featured = false,
  children,
}: {
  title: string;
  component: string;
  featured?: boolean;
  children: React.ReactNode;
}) {
  return (
    <article
      className={`ui-lab-production-panel ${featured ? 'ui-lab-production-panel-featured' : 'ui-lab-production-panel-tall'}`}
      data-editor-surface={component}
    >
      <header className="ui-lab-production-panel-label">
        <span>
          <strong>{title}</strong>
          <small>material editor surface</small>
        </span>
        <code>{component}</code>
      </header>
      <div className={`ui-lab-production-panel-stage ${featured ? 'ui-lab-material-graph-stage' : ''}`}>{children}</div>
    </article>
  );
}

export function UiLabMaterialControls() {
  return (
    <section className="ui-lab-section ui-lab-material-controls" aria-label="Material editor controls">
      <header className="ui-lab-section-header">
        <div>
          <h2>Material nodes</h2>
          <p>A production node card with representative value, pin, and parameter controls.</p>
        </div>
      </header>
      <div className="ui-lab-grid">
        <ControlCard title="Color node" caption="UiNodeCard / material controls">
          <UiLabMaterialNodeCard />
        </ControlCard>
      </div>
    </section>
  );
}

export function UiLabMaterialPanels() {
  return (
    <section className="ui-lab-material-panel-gallery" aria-label="Material editor surface gallery">
      <header className="ui-lab-panels-hero ui-lab-material-panels-hero">
        <div>
          <strong>Editor Surfaces</strong>
          <span>Production editor surfaces that are not registered as dockable workbench panels.</span>
        </div>
        <div className="ui-lab-panels-meta">
          <span>3 material surfaces</span>
          <span>Production components</span>
          <span>No native renderer required</span>
        </div>
      </header>
      <div className="ui-lab-panels-grid">
        <SurfaceCard featured title="Material Graph" component="MaterialGraphEditor">
          <UiLabMaterialGraph />
        </SurfaceCard>
        <SurfaceCard title="Material Preview" component="AssetPreviewPanel">
          <UiLabMaterialPreview />
        </SurfaceCard>
        <SurfaceCard title="Material Parameters" component="material-parameters-panel">
          <UiLabMaterialParameters />
        </SurfaceCard>
      </div>
    </section>
  );
}
