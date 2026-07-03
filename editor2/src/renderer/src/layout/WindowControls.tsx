import { useEffect, useState } from 'react';

export function WindowControls() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    void (window.arc?.nativeWindow?.isMaximized?.() ?? Promise.resolve(false)).then(setMaximized);
    return window.arc?.nativeWindow?.onMaximizedChanged?.(setMaximized);
  }, []);

  return (
    <div className="window-controls" aria-label="Window controls">
      <button title="Minimize" aria-label="Minimize" onClick={() => void window.arc?.nativeWindow?.minimize?.()}>
        <span className="window-control-icon window-control-minimize" aria-hidden="true" />
      </button>
      <button title={maximized ? 'Restore' : 'Maximize'} aria-label={maximized ? 'Restore' : 'Maximize'} onClick={() => void window.arc?.nativeWindow?.toggleMaximize?.()}>
        <span className={maximized ? 'window-control-icon window-control-restore' : 'window-control-icon window-control-maximize'} aria-hidden="true" />
      </button>
      <button className="window-control-exit" title="Close" aria-label="Close" onClick={() => void window.arc?.nativeWindow?.close?.()}>
        <span className="window-control-icon window-control-close" aria-hidden="true" />
      </button>
    </div>
  );
}
