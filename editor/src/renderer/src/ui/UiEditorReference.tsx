import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Box, FileBox, Image, Layers3 } from 'lucide-react';

import { useEditorReferenceController } from '../services/EditorReferenceContext';
import { parseEditorReference, type ResolvedEditorReference } from '../services/editorReferences';

import './uiEditorReference.css';

export type UiEditorReferenceProps = {
  href: string;
  children?: ReactNode;
  className?: string;
};

const ReferenceIcon = ({ kind }: { kind: 'entity' | 'asset' | 'scene' }) => {
  if (kind === 'entity') return <Box size={12} aria-hidden="true" />;
  if (kind === 'scene') return <Layers3 size={12} aria-hidden="true" />;
  return <FileBox size={12} aria-hidden="true" />;
};

export function UiEditorReference({ href, children, className }: UiEditorReferenceProps) {
  const controller = useEditorReferenceController();
  const reference = useMemo(() => parseEditorReference(href), [href]);
  const [resolved, setResolved] = useState<ResolvedEditorReference | null>(null);

  useEffect(() => {
    let cancelled = false;
    setResolved(null);
    if (!reference || !controller) return () => undefined;

    void Promise.resolve(controller.resolve(reference)).then((value) => {
      if (!cancelled) setResolved(value);
    });
    return () => {
      cancelled = true;
    };
  }, [controller, reference]);

  if (!reference) return <span className={className}>{children ?? href}</span>;

  const label = children ?? resolved?.label ?? reference.id;
  const classes = ['ui-editor-reference', className, resolved?.disabled ? 'is-disabled' : '']
    .filter(Boolean)
    .join(' ');

  return (
    <button
      className={classes}
      type="button"
      disabled={!controller || resolved?.disabled}
      title={resolved?.subtitle ?? `${reference.kind} reference`}
      aria-label={`${reference.kind} reference: ${resolved?.label ?? reference.id}`}
      onClick={() => {
        if (controller) void controller.activate(reference);
      }}
      onDoubleClick={() => {
        if (controller?.focus) void controller.focus(reference);
      }}
      onMouseEnter={() => {
        if (controller?.highlight) void controller.highlight(reference, true);
      }}
      onMouseLeave={() => {
        if (controller?.highlight) void controller.highlight(reference, false);
      }}
    >
      {resolved?.thumbnailUrl ? (
        <img className="ui-editor-reference-thumbnail" src={resolved.thumbnailUrl} alt="" aria-hidden="true" />
      ) : (
        <span className="ui-editor-reference-icon">
          <ReferenceIcon kind={reference.kind} />
        </span>
      )}
      <span className="ui-editor-reference-copy">
        <span className="ui-editor-reference-label">{label}</span>
        {resolved?.subtitle && <span className="ui-editor-reference-subtitle">{resolved.subtitle}</span>}
      </span>
    </button>
  );
}
