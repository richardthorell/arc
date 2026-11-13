// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor, act } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import { projectSettingsCatalog, type ProjectSettingsSnapshot } from '../../../common/projectSettingsTypes';
import { ProjectSettingsDialog } from './ProjectSettingsDialog';

let snapshot: ProjectSettingsSnapshot;
let notify: () => void;
const read = vi.fn();
const apply = vi.fn();
beforeEach(() => {
  snapshot = {
    catalog: structuredClone(projectSettingsCatalog),
    projectGuid: 'project',
    revision: '1',
    writable: true,
    values: Object.fromEntries(
      projectSettingsCatalog.fields.map((field) => [field.id, 'defaultValue' in field ? field.defaultValue : '']),
    ),
    capabilities: {
      version: 1,
      targets: ['windows-x64-vulkan', 'android-arm64-vulkan'],
      textureOutputs: ['bc', 'astc'],
      textureQualities: ['balanced'],
      androidMinimumSdk: 26,
      androidCompileSdk: 35,
    },
    configurations: ['Debug', 'Shipping'],
  };
  snapshot.values.name = 'Game';
  snapshot.values['package.applicationName'] = 'Game';
  snapshot.values.targetPlatforms = [
    { id: 'windows-x64-vulkan', enabled: true },
    { id: 'android-arm64-vulkan', enabled: true },
  ];
  read.mockReset().mockImplementation(async () => structuredClone(snapshot));
  apply.mockReset().mockImplementation(async (request) => {
    snapshot = { ...snapshot, revision: '2', values: { ...snapshot.values, ...request.changes } };
    return { succeeded: true, snapshot: structuredClone(snapshot) };
  });
  vi.stubGlobal('arc', {
    settings: {
      snapshot: vi.fn().mockResolvedValue({ pathValidation: {} }),
      project: {
        read,
        apply,
        onChanged: (listener: () => void) => {
          notify = listener;
          return vi.fn();
        },
      },
    },
    host: { query: vi.fn().mockResolvedValue({ payload: { assets: [] } }) },
  });
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it('keeps missing scene GUID bindings visible and unchanged when editing another setting', async () => {
  snapshot.values.defaultScene = { guid: 'missing-scene', expectedType: 'scene', pathHint: 'Content/Old.arcscene' };
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: /Startup/ }));
  expect(await screen.findByText(/Scene unavailable: Content\/Old.arcscene/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('treeitem', { name: /General/ }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Project name' }), { target: { value: 'Changed' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(apply).toHaveBeenCalled());
  expect(apply.mock.calls[0][0].changes).toEqual({ name: 'Changed' });
  expect(snapshot.values.defaultScene).toMatchObject({ guid: 'missing-scene', pathHint: 'Content/Old.arcscene' });
});

it('renders catalog categories and applies only edited fields with captured authority', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  const name = await screen.findByRole('textbox', { name: 'Project name' });
  expect(screen.getByRole('treeitem', { name: 'Identity & Metadata' })).toBeInTheDocument();
  expect(screen.getByRole('treeitem', { name: 'Android' })).toBeInTheDocument();
  fireEvent.change(name, { target: { value: 'Updated' } });
  expect(apply).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() =>
    expect(apply).toHaveBeenCalledWith({
      catalogVersion: 1,
      projectGuid: 'project',
      revision: '1',
      changes: { name: 'Updated' },
    }),
  );
  await waitFor(() => expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled());
});
it('uses preference-style rows and controls without a permanent footer label', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  const input = await screen.findByRole('textbox', { name: 'Project name' });
  expect(input).toHaveClass('settings-value-control');
  expect(input.closest('.settings-field-row')).not.toBeNull();
  expect(screen.queryByText('Project files are saved only when you Apply.')).not.toBeInTheDocument();
});

it('renders an additional catalog field without dialog-specific code and searches it', async () => {
  snapshot.catalog.fields.push({
    id: 'extension.note',
    category: 'general',
    label: 'Additional note',
    control: 'text',
  });
  snapshot.values['extension.note'] = 'visible';
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  expect(await screen.findByRole('textbox', { name: 'Additional note' })).toHaveValue('visible');
  fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'Additional' } });
  expect(screen.queryByRole('textbox', { name: 'Project name' })).not.toBeInTheDocument();
});

it('separates Android settings and exposes planned pages without editable placeholder settings', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'Application' }));
  expect(screen.getByRole('textbox', { name: 'Android application ID' })).toBeInTheDocument();
  expect(screen.queryByRole('spinbutton', { name: 'Android minimum SDK' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('treeitem', { name: 'Compatibility & Display' }));
  expect(screen.getByRole('spinbutton', { name: 'Android minimum SDK' })).toBeInTheDocument();
  fireEvent.click(screen.getByRole('treeitem', { name: 'Physics (Planned)' }));
  expect(screen.getByText(/Project-level physics authoring is planned/)).toBeInTheDocument();
  expect(screen.queryByRole('spinbutton')).not.toBeInTheDocument();
  expect(apply).not.toHaveBeenCalled();
});

it('displays existing project paths read-only without introducing relocation writes', async () => {
  snapshot.values['paths.content'] = 'GameContent';
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'Project Paths' }));
  expect(screen.getByText('GameContent')).toBeInTheDocument();
  expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
});
it('confirms discarding drafts and does not save them on close', async () => {
  const close = vi.fn();
  render(<ProjectSettingsDialog onClose={close} />);
  fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'Draft' } });
  fireEvent.keyDown(window, { key: 'Escape' });
  expect(close).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Discard changes' }));
  expect(close).toHaveBeenCalledOnce();
  expect(apply).not.toHaveBeenCalled();
});
it('refreshes clean settings but preserves a dirty draft on external changes', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  snapshot.revision = 'external1';
  snapshot.values.name = 'External';
  await act(async () => {
    notify();
  });
  expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('External');
  fireEvent.change(screen.getByRole('textbox', { name: 'Project name' }), { target: { value: 'Draft' } });
  snapshot.revision = 'external2';
  snapshot.values.name = 'External again';
  await act(async () => {
    notify();
  });
  expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('Draft');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Reload and discard draft' }));
  expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('External again');
});
it('retains drafts and shows native validation errors', async () => {
  apply.mockResolvedValue({ succeeded: false, errors: { _form: 'Invalid Android application ID' } });
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  fireEvent.change(await screen.findByRole('textbox', { name: 'Project name' }), { target: { value: 'Draft' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  expect(await screen.findByRole('alert')).toHaveTextContent('Invalid Android application ID');
  expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('Draft');
});
it('edits targets and profiles without implicitly deleting disabled-target profiles', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'Cooking' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add profile' }));
  expect(screen.getByRole('textbox', { name: 'Profile 1 ID' })).toHaveValue('profile');
  fireEvent.click(screen.getByRole('button', { name: 'Duplicate profile' }));
  expect(screen.getByRole('textbox', { name: 'Profile 2 ID' })).toHaveValue('profile-copy');
  fireEvent.click(screen.getByRole('treeitem', { name: 'Supported Platforms' }));
  fireEvent.click(screen.getByRole('switch', { name: 'windows-x64-vulkan' }));
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(apply).toHaveBeenCalled());
  expect(apply.mock.calls[0][0].changes.cookProfiles).toHaveLength(2);
});
it('disables authoring for a read-only project', async () => {
  snapshot.writable = false;
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  expect(await screen.findByRole('textbox', { name: 'Project name' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
});
