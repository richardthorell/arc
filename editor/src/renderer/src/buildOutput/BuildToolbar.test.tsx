// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ArcBuildSnapshot } from '../../../common/buildTypes';
import type { WorkflowExecutionSnapshot } from '../../../common/projectWorkflowTypes';
import { BuildToolbar } from './BuildToolbar';

const idle: ArcBuildSnapshot = {
  revision: 1,
  state: 'idle',
  configuration: 'RelWithDebInfo',
  buildRequired: false,
  reloadRequired: false,
  restartRequired: false,
  diagnostics: [],
};

const workflow: WorkflowExecutionSnapshot = {
  requestId: 'run-token',
  planning: true,
  run: null,
  output: [],
  error: null,
};

const execute = vi.fn().mockResolvedValue({});
const cancelWorkflow = vi.fn().mockResolvedValue(undefined);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

function setup(snapshot: ArcBuildSnapshot | null = idle, active: WorkflowExecutionSnapshot | null = null) {
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: { build: { execute, cancelWorkflow } },
  });
  const onCreate = vi.fn();
  const result = render(<BuildToolbar snapshot={snapshot} workflow={active} onCreate={onCreate} />);
  return { onCreate, ...result };
}

describe('Builds-only action toolbar', () => {
  it('uses shared toolbar regions and positions Build/Stop on the left', () => {
    const { onCreate, rerender } = setup();
    const toolbar = screen.getByRole('toolbar', { name: 'Builds toolbar' });
    expect(toolbar).toHaveClass('main-toolbar', 'ui-editor-toolbar', 'builds-editor-toolbar');
    expect(toolbar.querySelectorAll('[data-toolbar-region]')).toHaveLength(3);
    expect(toolbar.querySelector('[data-toolbar-region="left"]')).toContainElement(
      screen.getByRole('button', { name: 'Create build' }),
    );

    rerender(<BuildToolbar snapshot={idle} workflow={workflow} onCreate={onCreate} launchError="Build failed" />);
    expect(toolbar.querySelector('[data-toolbar-region="left"]')).toContainElement(
      screen.getByRole('button', { name: 'Stop' }),
    );
    expect(toolbar.querySelector('[data-toolbar-region="center"]')).toContainElement(screen.getByRole('status'));
    expect(toolbar.querySelector('[data-toolbar-region="right"]')).toContainElement(screen.getByRole('alert'));
  });

  it('opens Build... from its primary button and distinct operation dialogs from dropdown', () => {
    const { onCreate } = setup();
    fireEvent.click(screen.getByRole('button', { name: 'Create build' }));
    expect(onCreate).toHaveBeenCalledWith('build');

    fireEvent.click(screen.getByRole('button', { name: 'Build operations' }));
    fireEvent.click(screen.getByRole('menuitem', { name: 'Cook content...' }));
    expect(onCreate).toHaveBeenCalledWith('cook');

    fireEvent.click(screen.getByRole('button', { name: 'Build operations' }));
    expect(screen.getByRole('menuitem', { name: /Package separately/ })).toBeDisabled();
    fireEvent.click(screen.getByRole('menuitem', { name: 'Rebuild editor code...' }));
    expect(onCreate).toHaveBeenCalledWith('rebuild');
  });

  it('switches to red Stop for a running workflow and cancels only its request ID', async () => {
    const { onCreate, rerender } = setup(idle, workflow);
    expect(screen.queryByRole('button', { name: 'Create build' })).not.toBeInTheDocument();
    const stop = screen.getByRole('button', { name: 'Stop' });
    expect(stop).toHaveClass('builds-toolbar-stop');
    fireEvent.click(stop);
    await waitFor(() => expect(cancelWorkflow).toHaveBeenCalledWith('run-token'));
    expect(onCreate).not.toHaveBeenCalled();
    rerender(<BuildToolbar snapshot={idle} workflow={null} onCreate={onCreate} />);
    expect(screen.getByRole('button', { name: 'Create build' })).toBeInTheDocument();
  });

  it('displays an asynchronous Build launch error after the configuration dialog closes', () => {
    const { rerender, onCreate } = setup();
    rerender(<BuildToolbar snapshot={idle} workflow={null} onCreate={onCreate} launchError="SDK changed" />);
    expect(screen.getByRole('alert')).toHaveTextContent('SDK changed');
  });

  it('cancels the legacy command build only when it is actually active', async () => {
    setup({ ...idle, state: 'building' });
    fireEvent.click(screen.getByRole('button', { name: 'Stop' }));
    await waitFor(() => expect(execute).toHaveBeenCalledWith({ action: 'cancel' }));
    expect(cancelWorkflow).not.toHaveBeenCalled();
  });
});
