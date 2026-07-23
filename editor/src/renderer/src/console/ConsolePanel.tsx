import { useLayoutEffect, useRef } from 'react';
import { Lock, LockOpen, Trash2 } from 'lucide-react';

import type { ConsoleEvent } from '../services/mockHost';

type ConsolePanelProps = {
  events: ReadonlyArray<ConsoleEvent>;
  clearedIds: ReadonlySet<string>;
  locked: boolean;
  onClear: (events: ReadonlyArray<ConsoleEvent>) => void;
  onLockedChange: (locked: boolean) => void;
};

export function ConsolePanel({ events, clearedIds, locked, onClear, onLockedChange }: ConsolePanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const visibleEvents = events.filter((event) => !clearedIds.has(event.id));

  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (!locked || !element) return;
    element.scrollTop = element.scrollHeight;
  }, [locked, visibleEvents.length]);

  const unlockForManualScroll = () => {
    if (locked)
      onLockedChange(false);
  };

  return (
    <section className="console-panel" aria-label="Console">
      <div className="console-toolbar">
        <button aria-label="Clear console" disabled={visibleEvents.length === 0} onClick={() => onClear(events)} type="button">
          <Trash2 size={14} /> Clear
        </button>
        <button aria-label={locked ? 'Disable console scroll lock' : 'Enable console scroll lock'}
          aria-pressed={locked} className={locked ? 'active' : ''}
          onClick={() => onLockedChange(!locked)} type="button">
          {locked ? <Lock size={14} /> : <LockOpen size={14} />} Lock
        </button>
      </div>
      <div className="bottom-content console-content" onKeyDown={unlockForManualScroll}
        onPointerDown={unlockForManualScroll} onWheel={unlockForManualScroll} ref={scrollRef} tabIndex={0}>
        {visibleEvents.length === 0
          ? <div className="console-empty">Console is clear. New messages will appear here.</div>
          : visibleEvents.map((event) => (
            <div className={`log-line ${event.level}`} key={event.id}>
              [{event.timestamp}] [{event.source}] {event.message}
            </div>
          ))}
      </div>
    </section>
  );
}
