import { UiStatusBar } from '../layout/UiStatusBar';

import './UiLabStatusBar.css';

export function UiLabStatusBar() {
  return (
    <section className="ui-lab-section">
      <header className="ui-lab-section-header">
        <div>
          <h2>Status bar</h2>
          <p>Production editor status bar with a fixed in-progress activity preview for visual inspection.</p>
        </div>
      </header>
      <div className="ui-lab-grid">
        <article className="ui-lab-card ui-lab-card-wide ui-lab-status-bar-card">
          <header>
            <strong>Activity progress</strong>
            <code>UiStatusBar</code>
          </header>
          <div className="ui-lab-card-stage ui-lab-status-bar-stage">
            <UiStatusBar
              jobProgress={{ label: 'Loading assets', completed: 5, total: 10, indeterminate: false }}
              lastCommand=""
              startupState={null}
            />
          </div>
        </article>
      </div>
    </section>
  );
}
