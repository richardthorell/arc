// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { CookWorkflowOptions, WorkflowExecutionSnapshot } from '../../../common/projectWorkflowTypes';
import { CookWorkflowDialog } from './CookWorkflowDialog';
const options: CookWorkflowOptions = {
  authority: 'captured',
  profiles: [
    {
      id: 'mobile-custom',
      platform: 'android',
      targetId: 'android-arm64-vulkan',
      label: 'Mobile development / arm64 / vulkan / astc',
    },
  ],
};
const dirty = { id: 'material:a', kind: 'material' as const, title: 'Dirty Material', dirty: true, readOnly: false };
let notify: (value: WorkflowExecutionSnapshot) => void;
const execute = vi.fn();
const cancel = vi.fn();
const list = vi.fn();
beforeEach(() => {
  execute.mockResolvedValue({ state: 'succeeded' });
  cancel.mockResolvedValue(undefined);
  list.mockResolvedValue(options);
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      build: {
        cookOptions: list,
        executeWorkflow: execute,
        cancelWorkflow: cancel,
        workflowSnapshot: vi.fn().mockResolvedValue(null),
        onWorkflowState: (callback: typeof notify) => {
          notify = callback;
          return vi.fn();
        },
      },
    },
  });
});
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
function open(saveDocument = vi.fn().mockResolvedValue(true), onClose = vi.fn()) {
  render(<CookWorkflowDialog platform="android" documents={[dirty]} saveDocument={saveDocument} onClose={onClose} />);
  return { saveDocument, onClose };
}
async function choose(label: string) {
  await waitFor(() => expect(screen.getByRole('button', { name: label })).toBeEnabled());
  fireEvent.click(screen.getByRole('button', { name: label }));
}
it('cooks the explicit profile and captured authority without saving dirty content', async () => {
  const { saveDocument } = open();
  await choose('Cook Saved Content');
  await waitFor(() =>
    expect(execute).toHaveBeenCalledWith(
      {
        action: 'cook',
        profile: 'mobile-custom',
        targetId: 'android-arm64-vulkan',
        destination: 'none',
        sessionId: '',
      },
      'captured',
    ),
  );
  expect(saveDocument).not.toHaveBeenCalled();
});
it('saves only after the explicit choice and starts one cook', async () => {
  const { saveDocument } = open();
  await choose('Save All and Cook');
  await waitFor(() => expect(execute).toHaveBeenCalledTimes(1));
  expect(saveDocument).toHaveBeenCalledWith(dirty);
});
it('does not cook after a cancelled or failed save', async () => {
  open(vi.fn().mockResolvedValue(false));
  await choose('Save All and Cook');
  expect(await screen.findByRole('alert')).toHaveTextContent('Save cancelled or failed');
  expect(execute).not.toHaveBeenCalled();
});
it('rejects changed project authority before saving', async () => {
  const { saveDocument } = open();
  await waitFor(() => expect(screen.getByRole('button', { name: 'Save All and Cook' })).toBeEnabled());
  list.mockResolvedValue({ ...options, authority: 'replacement' });
  fireEvent.click(screen.getByRole('button', { name: 'Save All and Cook' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Project or SDK changed');
  expect(saveDocument).not.toHaveBeenCalled();
  expect(execute).not.toHaveBeenCalled();
});
it('cancel dismisses without saving or starting work', async () => {
  const { onClose, saveDocument } = open();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(onClose).toHaveBeenCalledOnce();
  expect(saveDocument).not.toHaveBeenCalled();
  expect(execute).not.toHaveBeenCalled();
});
it('keeps profiles explicit when several are available', async () => {
  list.mockResolvedValue({ ...options, profiles: [...options.profiles, { ...options.profiles[0], id: 'shipping' }] });
  open();
  await screen.findByText('Select a profile');
  expect(screen.getByRole('button', { name: 'Cook Saved Content' })).toBeDisabled();
});
it('shows missing cooker diagnostics without allowing execution', async () => {
  list.mockResolvedValue({ ...options, unavailableReason: 'ARC cooker is unavailable in the selected SDK' });
  open();
  expect(await screen.findByRole('alert')).toHaveTextContent('cooker is unavailable');
  expect(screen.getByRole('button', { name: 'Cook Saved Content' })).toBeDisabled();
});
it('shows progress and cancels only the captured request', async () => {
  open();
  await act(async () => {
    notify({ requestId: 'owned-request', planning: true, run: null, output: ['Preparing'], error: null });
  });
  expect(screen.getByText('Planning')).toBeInTheDocument();
  expect(screen.getByText('Preparing')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Cancel Cook' }));
  expect(cancel).toHaveBeenCalledWith('owned-request');
  expect(execute).not.toHaveBeenCalled();
});

it('does not let an old initial snapshot overwrite a newer progress event', async () => {
  let resolve!: (value: WorkflowExecutionSnapshot | null) => void;
  vi.mocked(window.arc.build.workflowSnapshot).mockReturnValue(
    new Promise((done) => {
      resolve = done;
    }),
  );
  open();
  await act(async () => {
    notify({ requestId: 'current', planning: true, run: null, output: [], error: null });
    resolve(null);
  });
  expect(screen.getByText('Planning')).toBeInTheDocument();
});

it('ignores profile results from the previous platform', async () => {
  let resolve!: (value: CookWorkflowOptions) => void;
  list.mockReturnValueOnce(
    new Promise((done) => {
      resolve = done;
    }),
  );
  const props = { documents: [], saveDocument: vi.fn(), onClose: vi.fn() };
  const view = render(<CookWorkflowDialog {...props} platform="windows" />);
  view.rerender(<CookWorkflowDialog {...props} platform="android" />);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Cook Content' })).toBeEnabled());
  await act(async () => {
    resolve({ authority: 'old', profiles: [] });
  });
  expect(screen.getByRole('button', { name: 'Cook Content' })).toBeEnabled();
  fireEvent.click(screen.getByRole('button', { name: 'Cook Content' }));
  await waitFor(() =>
    expect(execute).toHaveBeenCalledWith(expect.objectContaining({ profile: 'mobile-custom' }), 'captured'),
  );
});

it('does not start a cook after the dialog closes during a save', async () => {
  let resolve!: (value: boolean) => void;
  const save = vi.fn(
    () =>
      new Promise<boolean>((done) => {
        resolve = done;
      }),
  );
  const view = render(
    <CookWorkflowDialog platform="android" documents={[dirty]} saveDocument={save} onClose={vi.fn()} />,
  );
  await choose('Save All and Cook');
  await waitFor(() => expect(save).toHaveBeenCalled());
  view.unmount();
  await act(async () => {
    resolve(true);
  });
  expect(execute).not.toHaveBeenCalled();
});
