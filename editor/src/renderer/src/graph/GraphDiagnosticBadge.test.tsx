// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import { GraphDiagnosticBadge } from './GraphDiagnosticBadge';

it('shows shared graph diagnostic details and activates its navigation target', () => {
  const onActivate = vi.fn();
  render(
    <GraphDiagnosticBadge
      onActivate={onActivate}
      summary={{
        nodeId: 'node-a',
        highestSeverity: 'warning',
        count: 2,
        errorCount: 0,
        warningCount: 2,
        infoCount: 0,
        diagnostics: [
          {
            id: 'a',
            severity: 'warning',
            message: 'First warning',
            target: { kind: 'node', nodeId: 'node-a' },
          },
          {
            id: 'b',
            severity: 'warning',
            message: 'Second warning',
            target: { kind: 'node', nodeId: 'node-a' },
          },
        ],
      }}
    />,
  );

  const button = screen.getByRole('button', { name: 'Warning: 2 graph diagnostics' });
  expect(screen.getByRole('tooltip')).toHaveTextContent('First warning');
  expect(screen.getByRole('tooltip')).toHaveTextContent('Second warning');
  fireEvent.click(button);
  expect(onActivate).toHaveBeenCalledTimes(1);
});
