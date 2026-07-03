import { useEffect, useState } from 'react';
import { Maximize2, Minimize2, Square } from 'lucide-react';

export function WindowControls() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    void window.arc?.nativeWindow?.isMaximized?.().then(setMaximized);
    return window.arc?.nativeWindow?.onMaximizedChanged?.(setMaximized);
  }, []);

  return (
    <div className="window-controls" aria-label="Window controls">
      <button title="Minimize" onClick={() => void window.arc?.nativeWindow?.minimize?.()}>
        <Minimize2 size={13} />
      </button>
      <button title={maximized ? 'Restore' : 'Maximize'} onClick={() => void window.arc?.nativeWindow?.toggleMaximize?.()}>
        {maximized ? <Square size={12} /> : <Maximize2 size={12} />}
      </button>
      <button className="window-control-exit" title="Exit" onClick={() => void window.arc?.nativeWindow?.close?.()}>
        ×
      </button>
    </div>
  );
}
