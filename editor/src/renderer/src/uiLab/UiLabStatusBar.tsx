import { StatusBar } from '../layout/StatusBar';

import './UiLabStatusBar.css';

export function UiLabStatusBar() {
  return (
    <section className="ui-lab-section">
      <header className="ui-lab-section-header">
        <div>
          <h2>Status bar</h2>
          <p>Production editor status bar with a fixed in-progress job preview for visual inspection.</p>
        </div>
      </header>
      <div className="ui-lab-grid">
        <article className="ui-lab-card ui-lab-card-wide">
          <header>
            <strong>Job progress</strong>
            <code>StatusBar</code>
          </header>
          <div className="ui-lab-card-stage ui-lab-status-bar-stage">
            <StatusBar jobProgress={{ completed: 5, total: 10 }} lastCommand="" startupState={null} />
          </div>
        </article>
      </div>
    </section>
  );
}
