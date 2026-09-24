import { Search } from 'lucide-react';

import { UiButton } from './UiButton';
import { UiSearchInput } from './UiTextInput';

import './UiSearchHeader.css';

export type UiSearchHeaderMode = {
  id: string;
  label: string;
  count?: number;
};

export type UiSearchHeaderProps = {
  title?: string;
  query: string;
  mode: string;
  modes: readonly UiSearchHeaderMode[];
  resultCount?: number;
  placeholder?: string;
  searchLabel?: string;
  onQueryChange: (query: string) => void;
  onModeChange: (mode: string) => void;
};

export function UiSearchHeader({
  title = 'Search',
  query,
  mode,
  modes,
  resultCount,
  placeholder = 'Search…',
  searchLabel = 'Search',
  onQueryChange,
  onModeChange,
}: UiSearchHeaderProps) {
  return (
    <header className="ui-search-header">
      <div className="ui-search-header-topline">
        <div className="ui-search-header-title">
          <strong>{title}</strong>
          {resultCount !== undefined && <small>{resultCount} results</small>}
        </div>
        <div aria-label={`${title} scope`} className="ui-search-header-modes" role="tablist">
          {modes.map((entry) => {
            const active = entry.id === mode;
            return (
              <UiButton
                active={active}
                aria-selected={active}
                className="ui-search-header-mode"
                key={entry.id}
                role="tab"
                type="button"
                variant="ghost"
                onClick={() => onModeChange(entry.id)}
              >
                <span>{entry.label}</span>
                {entry.count !== undefined && <small>{entry.count}</small>}
              </UiButton>
            );
          })}
        </div>
      </div>
      <label className="ui-search-header-field">
        <Search aria-hidden="true" size={15} />
        <UiSearchInput
          aria-label={searchLabel}
          placeholder={placeholder}
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
        />
      </label>
    </header>
  );
}
