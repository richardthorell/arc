import { Check, FileBox } from 'lucide-react';
import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { useEditorReferenceController } from '../services/EditorReferenceContext';
import { parseEditorReference, type ResolvedEditorReference } from '../services/editorReferences';
import { UiAgentCard } from './UiAgentCard';
import { UiButton } from './UiButton';

import './UiAgentChoiceCard.css';

export type UiAgentChoiceCardProps = {
  title: string;
  prompt?: string;
  children: ReactNode;
  confirmDisabled?: boolean;
  disabled?: boolean;
  confirmLabel?: string;
  onConfirm: () => void;
};

export function UiAgentChoiceCard({
  title,
  prompt,
  children,
  confirmDisabled = false,
  disabled = false,
  confirmLabel = 'Use this',
  onConfirm,
}: UiAgentChoiceCardProps) {
  return (
    <UiAgentCard
      className="ui-agent-choice-card"
      side="none"
      subtitle="Choose one"
      title={title}
      tone="neutral"
    >
      {prompt ? <p className="ui-agent-choice-prompt">{prompt}</p> : null}
      {children}
      <div className="ui-agent-choice-actions">
        <UiButton disabled={disabled || confirmDisabled} type="button" variant="primary" onClick={onConfirm}>
          {confirmLabel}
        </UiButton>
      </div>
    </UiAgentCard>
  );
}

export type UiAgentAssetChoiceOption = {
  uri: string;
  label: string;
  reason?: string;
};

export type UiAgentAssetChoiceCardProps = {
  title: string;
  prompt?: string;
  options: readonly UiAgentAssetChoiceOption[];
  disabled?: boolean;
  onChoose: (uri: string) => void;
};

function AssetChoiceOption({
  option,
  selected,
  disabled,
  onSelect,
}: {
  option: UiAgentAssetChoiceOption;
  selected: boolean;
  disabled?: boolean;
  onSelect: () => void;
}) {
  const controller = useEditorReferenceController();
  const reference = useMemo(() => parseEditorReference(option.uri), [option.uri]);
  const [resolved, setResolved] = useState<ResolvedEditorReference | null | undefined>(undefined);

  useEffect(() => {
    let cancelled = false;
    setResolved(undefined);
    if (!reference || reference.kind !== 'asset' || !controller) {
      setResolved(null);
      return () => undefined;
    }

    void Promise.resolve(controller.resolve(reference)).then((value) => {
      if (!cancelled) setResolved(value);
    });
    return () => {
      cancelled = true;
    };
  }, [controller, option.uri, reference]);

  const unavailable =
    !controller || !reference || reference.kind !== 'asset' || resolved === undefined || resolved === null || resolved.disabled;
  const label = resolved?.label ?? option.label;
  const subtitle = resolved?.subtitle ?? 'Asset';

  return (
    <button
      aria-label={`Choose ${label}`}
      aria-pressed={selected}
      className="ui-agent-asset-choice-option"
      data-selected={selected ? 'true' : 'false'}
      disabled={disabled || unavailable}
      type="button"
      onClick={onSelect}
      onDoubleClick={() => {
        if (reference && controller?.focus) void controller.focus(reference);
      }}
    >
      <span className="ui-agent-asset-choice-preview">
        {resolved?.thumbnailUrl ? (
          <img src={resolved.thumbnailUrl} alt="" aria-hidden="true" />
        ) : (
          <FileBox aria-hidden="true" size={24} />
        )}
        {selected ? (
          <span className="ui-agent-asset-choice-check" aria-hidden="true">
            <Check size={12} />
          </span>
        ) : null}
      </span>
      <span className="ui-agent-asset-choice-copy">
        <strong>{label}</strong>
        <small>{subtitle}</small>
        {option.reason ? <span>{option.reason}</span> : null}
      </span>
    </button>
  );
}

export function UiAgentAssetChoiceCard({
  title,
  prompt,
  options,
  disabled = false,
  onChoose,
}: UiAgentAssetChoiceCardProps) {
  const [selectedUri, setSelectedUri] = useState<string | null>(null);
  const [confirmedUri, setConfirmedUri] = useState<string | null>(null);
  const selected = options.find((option) => option.uri === selectedUri) ?? null;
  const confirmed = confirmedUri !== null;

  useEffect(() => {
    if (selectedUri && !options.some((option) => option.uri === selectedUri)) setSelectedUri(null);
    if (confirmedUri && !options.some((option) => option.uri === confirmedUri)) setConfirmedUri(null);
  }, [confirmedUri, options, selectedUri]);

  return (
    <div data-agent-choice-kind="asset">
      <UiAgentChoiceCard
        confirmDisabled={!selected || confirmed}
        confirmLabel={confirmed ? 'Selected' : 'Use this'}
        disabled={disabled}
        prompt={prompt}
        title={title}
        onConfirm={() => {
          if (!selected || confirmed) return;
          setConfirmedUri(selected.uri);
          onChoose(selected.uri);
        }}
      >
        <div className="ui-agent-asset-choice-grid">
          {options.map((option) => (
            <AssetChoiceOption
              disabled={disabled || confirmed}
              key={option.uri}
              option={option}
              selected={option.uri === selectedUri}
              onSelect={() => setSelectedUri(option.uri)}
            />
          ))}
        </div>
      </UiAgentChoiceCard>
    </div>
  );
}
