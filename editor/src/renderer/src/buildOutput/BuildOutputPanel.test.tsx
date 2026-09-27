// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { BuildOutputPanel } from './BuildOutputPanel';

describe('BuildOutputPanel', () => {
  it('launches build actions for the active configuration and opens source diagnostics', () => {
    const execute = vi.fn();
    const openDiagnostic = vi.fn();
    const diagnostic = {
      sequence: 1,
      severity: 'error' as const,
      message: 'Unknown identifier',
      file: 'Source/Game/Component.cpp',
      line: 42,
      column: 7,
      category: 'compiler' as const,
    };
    render(
      <BuildOutputPanel
        snapshot={{
          revision: 2,
          state: 'failed',
          configuration: 'RelWithDebInfo',
          buildRequired: true,
          reloadRequired: true,
          restartRequired: false,
          diagnostics: [diagnostic],
        }}
        onExecute={execute}
        onOpenDiagnostic={openDiagnostic}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Build' }));
    expect(execute).toHaveBeenCalledWith({ action: 'build', configuration: 'RelWithDebInfo' });
    fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
    expect(execute).toHaveBeenCalledWith({ action: 'reload' });
    fireEvent.click(screen.getByTitle('Source/Game/Component.cpp:42'));
    expect(openDiagnostic).toHaveBeenCalledWith(diagnostic);
  });

  it('uses the selected configuration for subsequent build actions', () => {
    const execute = vi.fn();
    render(<BuildOutputPanel snapshot={null} onExecute={execute} onOpenDiagnostic={vi.fn()} />);

    fireEvent.click(screen.getByRole('combobox', { name: 'Build configuration' }));
    fireEvent.click(screen.getByRole('option', { name: 'Release' }));
    fireEvent.click(screen.getByRole('button', { name: 'Rebuild' }));

    expect(execute).toHaveBeenCalledWith({ action: 'rebuild', configuration: 'Release' });
  });

  it('disables configuration changes while a build operation is active', () => {
    render(
      <BuildOutputPanel
        snapshot={{
          revision: 3,
          state: 'building',
          configuration: 'Debug',
          buildRequired: false,
          reloadRequired: false,
          restartRequired: false,
          diagnostics: [],
        }}
        onExecute={vi.fn()}
        onOpenDiagnostic={vi.fn()}
      />,
    );

    expect(screen.getByRole('combobox', { name: 'Build configuration' })).toBeDisabled();
  });
});
