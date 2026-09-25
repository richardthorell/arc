import { Command as CommandIcon, Database, SearchX } from 'lucide-react';

import './UiSearchList.css';

export type UiSearchListItem = {
  id: string;
  variant: 'asset' | 'command';
  title: string;
  subtitle?: string;
  meta?: string;
  state?: string;
  shortcut?: string;
  disabled?: boolean;
  disabledReason?: string;
};

export type UiSearchListProps<T extends UiSearchListItem = UiSearchListItem> = {
  items: readonly T[];
  onActivate: (item: T) => void;
  emptyMessage?: string;
  ariaLabel?: string;
};

const stateToken = (value: string) => value.toLocaleLowerCase().replace(/[^a-z0-9]+/g, '-');

export function UiSearchList<T extends UiSearchListItem>({
  items,
  onActivate,
  emptyMessage = 'No matching results',
  ariaLabel = 'Search results',
}: UiSearchListProps<T>) {
  if (items.length === 0) {
    return (
      <div className="ui-search-list-empty" role="status">
        <SearchX aria-hidden="true" size={20} />
        <span>{emptyMessage}</span>
      </div>
    );
  }

  return (
    <div aria-label={ariaLabel} className="ui-search-list" role="list">
      {items.map((item) => {
        const unavailable = item.disabled ?? false;
        const state = unavailable && item.variant === 'command' ? 'Unavailable' : item.state;
        return (
          <div className="ui-search-list-item" key={item.id} role="listitem">
            <button
              aria-label={item.title}
              className={['ui-search-list-row', unavailable ? 'is-disabled' : ''].filter(Boolean).join(' ')}
              disabled={unavailable}
              title={unavailable ? item.disabledReason : undefined}
              type="button"
              onClick={() => onActivate(item)}
            >
              <span aria-hidden="true" className={`ui-search-list-icon is-${item.variant}`}>
                {item.variant === 'asset' ? <Database size={16} /> : <CommandIcon size={16} />}
              </span>
              <span className="ui-search-list-copy">
                <strong>{item.title}</strong>
                {item.subtitle && <small>{item.subtitle}</small>}
              </span>
              <span className="ui-search-list-trailing">
                <span className="ui-search-list-badges">
                  {item.meta && <span className="ui-search-list-meta">{item.meta}</span>}
                  {state && (
                    <span className="ui-search-list-state" data-state={stateToken(state)}>
                      {state}
                    </span>
                  )}
                </span>
                {item.shortcut && <kbd>{item.shortcut}</kbd>}
              </span>
            </button>
          </div>
        );
      })}
    </div>
  );
}
