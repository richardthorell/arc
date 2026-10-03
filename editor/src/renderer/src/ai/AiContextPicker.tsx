import { Box, Bug, Camera, FileBox, Image, Monitor, MousePointer2, Package, Search, X } from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';

import type { AiConversationContextReference } from '../../../common/aiConversationTypes';
import type { AiProjectContextSnapshot } from '../../../common/aiContextTypes';
import { UiFloatingSurface, UiIconButton, UiSearchInput } from '../ui';
import type { AiProjectContextSource } from './aiContextBudget';
import {
  aiContextReferenceFromCandidate,
  captureAiViewportReference,
  collectAiContextPickerCandidates,
  type AiContextPickerCandidate,
} from './aiContextPicker';

import './aiContextPicker.css';

type AiContextPickerProps = {
  source: AiProjectContextSource | null | undefined;
  selected: readonly AiConversationContextReference[];
  supportsImages: boolean;
  onAdd: (reference: AiConversationContextReference) => void;
  onClose: () => void;
};

type AiContextChipsProps = {
  references: readonly AiConversationContextReference[];
  onRemove: (id: string) => void;
};

const iconForKind = (kind: string, size = 14): ReactNode => {
  if (kind === 'selection') return <MousePointer2 size={size} />;
  if (kind === 'scene') return <Box size={size} />;
  if (kind === 'workspace') return <FileBox size={size} />;
  if (kind === 'viewport') return <Monitor size={size} />;
  if (kind === 'viewportCapture') return <Image size={size} />;
  if (kind === 'diagnostics') return <Bug size={size} />;
  if (kind === 'entity') return <Box size={size} />;
  if (kind === 'asset') return <Package size={size} />;
  return <FileBox size={size} />;
};

const candidateMatches = (candidate: AiContextPickerCandidate, query: string) => {
  const normalized = query.trim().toLocaleLowerCase();
  if (!normalized) return true;
  return `${candidate.label} ${candidate.detail ?? ''} ${candidate.kind}`.toLocaleLowerCase().includes(normalized);
};

const candidateSection = (
  title: string,
  candidates: readonly AiContextPickerCandidate[],
  snapshot: AiProjectContextSnapshot,
  selectedIds: ReadonlySet<string>,
  onAdd: (reference: AiConversationContextReference) => void,
) => {
  if (!candidates.length) return null;
  return (
    <section className="ai-context-picker-section" aria-label={title}>
      <span className="ai-context-picker-section-title">{title}</span>
      <div className="ai-context-picker-list">
        {candidates.map((candidate) => {
          const selected = selectedIds.has(candidate.id);
          return (
            <button
              className="ai-context-picker-row"
              disabled={selected}
              key={candidate.id}
              type="button"
              onClick={() => onAdd(aiContextReferenceFromCandidate(snapshot, candidate))}
            >
              <span className="ai-context-picker-row-icon" aria-hidden="true">
                {iconForKind(candidate.kind, 15)}
              </span>
              <span className="ai-context-picker-row-copy">
                <strong>{candidate.label}</strong>
                {candidate.detail && <small>{candidate.detail}</small>}
              </span>
              {selected && <span className="ai-context-picker-added">Added</span>}
            </button>
          );
        })}
      </div>
    </section>
  );
};

export function AiContextPicker({ source, selected, supportsImages, onAdd, onClose }: AiContextPickerProps) {
  const [snapshot, setSnapshot] = useState<AiProjectContextSnapshot | null>(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    setLoading(true);
    setError(null);
    if (!source) {
      setLoading(false);
      setError('Project context is unavailable');
      return () => {
        disposed = true;
      };
    }
    void source
      .collect({ forceRefresh: true })
      .then((next) => {
        if (!disposed) setSnapshot(next);
      })
      .catch((reason) => {
        if (!disposed) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!disposed) setLoading(false);
      });
    return () => {
      disposed = true;
    };
  }, [source]);

  const candidates = useMemo(
    () =>
      snapshot
        ? collectAiContextPickerCandidates(snapshot).filter((candidate) => candidateMatches(candidate, query))
        : [],
    [query, snapshot],
  );
  const selectedIds = useMemo(() => new Set(selected.map((reference) => reference.id)), [selected]);
  const quick = candidates.filter((candidate) => candidate.kind !== 'entity' && candidate.kind !== 'asset');
  const entities = candidates.filter((candidate) => candidate.kind === 'entity');
  const assets = candidates.filter((candidate) => candidate.kind === 'asset');
  const viewportId = snapshot?.sections.find((section) => section.id === 'viewport')?.data;
  const captureSelected = selected.some((reference) => reference.kind === 'viewportCapture');

  const captureViewport = () => {
    if (!snapshot) return;
    try {
      onAdd(captureAiViewportReference(snapshot));
      setError(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  return (
    <UiFloatingSurface className="ai-context-picker" aria-label="Add context" role="dialog">
      <header className="ai-context-picker-header">
        <div>
          <strong>Add context</strong>
          <small>Attach editor state to the next message</small>
        </div>
        <UiIconButton label="Close context picker" type="button" variant="ghost" onClick={onClose}>
          <X size={14} />
        </UiIconButton>
      </header>
      <label className="ai-context-picker-search">
        <Search aria-hidden="true" size={14} />
        <UiSearchInput
          aria-label="Search context"
          autoFocus
          placeholder="Search entities and assets"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>

      <div className="ai-context-picker-scroll">
        {loading && <div className="ai-context-picker-status">Collecting fresh project context…</div>}
        {error && (
          <div className="ai-context-picker-error" role="alert">
            {error}
          </div>
        )}
        {!loading && snapshot && (
          <>
            {candidateSection('Current editor context', quick, snapshot, selectedIds, onAdd)}
            {!query.trim() && (
              <section className="ai-context-picker-section" aria-label="Viewport attachment">
                <span className="ai-context-picker-section-title">Viewport attachment</span>
                <div className="ai-context-picker-list">
                  <button
                    className="ai-context-picker-row"
                    disabled={!supportsImages || captureSelected}
                    title={!supportsImages ? 'The selected model does not accept image input' : undefined}
                    type="button"
                    onClick={captureViewport}
                  >
                    <span className="ai-context-picker-row-icon" aria-hidden="true">
                      <Camera size={15} />
                    </span>
                    <span className="ai-context-picker-row-copy">
                      <strong>Viewport capture</strong>
                      <small>
                        {supportsImages ? 'Attach the current rendered frame' : 'Requires an image-capable model'}
                      </small>
                    </span>
                    {captureSelected && <span className="ai-context-picker-added">Added</span>}
                  </button>
                </div>
              </section>
            )}
            {candidateSection('Entities', entities, snapshot, selectedIds, onAdd)}
            {candidateSection('Assets', assets, snapshot, selectedIds, onAdd)}
            {!candidates.length && query.trim() && (
              <div className="ai-context-picker-status">No matching project context</div>
            )}
          </>
        )}
      </div>
      {viewportId === undefined && !loading && snapshot && (
        <footer className="ai-context-picker-footer">Viewport context is unavailable for this editor state.</footer>
      )}
    </UiFloatingSurface>
  );
}

export function AiContextChips({ references, onRemove }: AiContextChipsProps) {
  if (!references.length) return null;
  return (
    <div className="ai-context-chips" aria-label="Attached context">
      {references.map((reference) => (
        <span className="ai-context-chip" key={reference.id}>
          <span aria-hidden="true">{iconForKind(reference.kind, 12)}</span>
          <span className="ai-context-chip-label">{reference.label ?? reference.kind}</span>
          <button
            aria-label={`Remove ${reference.label ?? reference.kind}`}
            type="button"
            onClick={() => onRemove(reference.id)}
          >
            <X size={11} />
          </button>
        </span>
      ))}
    </div>
  );
}
