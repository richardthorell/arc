import { AlertCircle, Info, TriangleAlert } from 'lucide-react';

import type { GraphDiagnosticSummary } from './graphDiagnostics';
import './graphDiagnostics.css';

const severityLabel = (severity: GraphDiagnosticSummary['highestSeverity']) =>
  severity === 'error' ? 'Error' : severity === 'warning' ? 'Warning' : 'Info';

export function GraphDiagnosticBadge({
  summary,
  onActivate,
}: {
  summary: GraphDiagnosticSummary;
  onActivate?: () => void;
}) {
  const Icon =
    summary.highestSeverity === 'error' ? AlertCircle : summary.highestSeverity === 'warning' ? TriangleAlert : Info;
  const label = `${severityLabel(summary.highestSeverity)}: ${summary.count} graph diagnostic${summary.count === 1 ? '' : 's'}`;

  return (
    <div className={`graph-diagnostic-badge is-${summary.highestSeverity}`}>
      <button
        aria-label={label}
        onClick={(event) => {
          event.stopPropagation();
          onActivate?.();
        }}
        onPointerDown={(event) => event.stopPropagation()}
        type="button"
      >
        <Icon aria-hidden="true" size={12} />
        {summary.count > 1 && <span>{summary.count}</span>}
      </button>
      <div className="graph-diagnostic-details" role="tooltip">
        <strong>{severityLabel(summary.highestSeverity)}</strong>
        {summary.diagnostics.map((diagnostic) => (
          <span key={diagnostic.id}>{diagnostic.message}</span>
        ))}
      </div>
    </div>
  );
}
