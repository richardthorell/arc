// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor, act } from '@testing-library/react';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import {
  projectSettingsCatalog,
  visibleProjectSettingsCategories,
  type ProjectSettingsSnapshot,
} from '../../../common/projectSettingsTypes';
import { ProjectSettingsDialog } from './ProjectSettingsDialog';
import {
  cacheProjectSettings,
  cachedProjectSettings,
  clearProjectSettingsCache,
  refreshProjectSettings,
} from './projectSettingsCache';

let snapshot: ProjectSettingsSnapshot;
let notify: () => void;
const read = vi.fn();
const apply = vi.fn();
beforeEach(() => {
  clearProjectSettingsCache();
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
  clearProjectSettingsCache();
  vi.unstubAllGlobals();
});

it('opens immediately from the project snapshot while refreshing in the background', () => {
  cacheProjectSettings(snapshot);
  read.mockImplementation(() => new Promise(() => {}));
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  expect(screen.getByRole('textbox', { name: 'Project name' })).toHaveValue('Game');
  expect(screen.queryByText('Loading project settings…')).not.toBeInTheDocument();
  expect(screen.queryByText('Local tool availability')).not.toBeInTheDocument();
});

it('does not show platform tool information during an uncached read', () => {
  read.mockImplementation(() => new Promise(() => {}));
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  expect(screen.queryByText('Local tool availability')).not.toBeInTheDocument();
});

it('does not restore a previous project cache after a late read', async () => {
  let resolve!: (value: ProjectSettingsSnapshot) => void;
  read.mockImplementation(
    () =>
      new Promise<ProjectSettingsSnapshot>((done) => {
        resolve = done;
      }),
  );
  const pending = refreshProjectSettings();
  clearProjectSettingsCache();
  resolve(snapshot);
  await expect(pending).rejects.toThrow('Project changed');
  expect(cachedProjectSettings()).toBeUndefined();
});

it('shows project information controls and keeps engine identity in Advanced', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  expect(screen.getByRole('textbox', { name: 'Description' })).toHaveAttribute('rows', '3');
  expect(screen.getByRole('textbox', { name: 'Studio / Company' })).toBeInTheDocument();
  expect(screen.getByRole('textbox', { name: 'Project version' })).toHaveValue('0.1.0');
  expect(screen.getByText('Project thumbnail')).toBeInTheDocument();
  expect(screen.queryByText('Application display name')).not.toBeInTheDocument();
  expect(screen.queryByText('Project GUID')).not.toBeInTheDocument();
  expect(screen.getByText('Project identity and default application name.')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('treeitem', { name: 'Advanced' }));
  expect(screen.getByText('Project GUID')).toBeInTheDocument();
  expect(screen.getByText('Engine version')).toBeInTheDocument();
});

it('keeps missing scene GUID bindings visible and unchanged when editing another setting', async () => {
  snapshot.values.startupScene = { guid: 'missing-scene', expectedType: 'scene', pathHint: 'Content/Old.arcscene' };
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'General' }));
  expect(await screen.findByText(/Scene unavailable: Content\/Old.arcscene/)).toBeInTheDocument();
  fireEvent.click(screen.getByRole('treeitem', { name: /General/ }));
  fireEvent.change(screen.getByRole('textbox', { name: 'Project name' }), { target: { value: 'Changed' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply' }));
  await waitFor(() => expect(apply).toHaveBeenCalled());
  expect(apply.mock.calls[0][0].changes).toEqual({ name: 'Changed' });
  expect(snapshot.values.startupScene).toMatchObject({ guid: 'missing-scene', pathHint: 'Content/Old.arcscene' });
});

it('shows a compact game picker and exposes server inheritance only for enabled server targets', async () => {
  const view = render(<ProjectSettingsDialog onClose={vi.fn()} />);
  const picker = await screen.findByRole('button', { name: 'Choose Startup scene asset' });
  expect(picker.querySelector('.asset-thumbnail')).toBeNull();
  expect(screen.getAllByText('Startup scene')).toHaveLength(1);
  expect(screen.queryByText('Editor default scene')).not.toBeInTheDocument();
  expect(screen.queryByText('Runtime startup scenes')).not.toBeInTheDocument();
  expect(screen.queryByText('Server startup scene')).not.toBeInTheDocument();
  expect(screen.queryByText('Settings shared by this project')).not.toBeInTheDocument();
  view.unmount();
  snapshot.values.targetPlatforms = [{ id: 'linux-x64-headless', enabled: true }];
  snapshot.revision = '2';
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  expect(await screen.findByText('Use game startup scene')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Choose Server startup scene asset' })).toBeInTheDocument();
});

it('renders catalog categories and applies only edited fields with captured authority', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  const name = await screen.findByRole('textbox', { name: 'Project name' });
  expect(screen.getByRole('treeitem', { name: 'General' })).toBeInTheDocument();
  expect(screen.queryByRole('treeitem', { name: /Overview|Startup|Project Information/ })).not.toBeInTheDocument();
  expect(screen.getByText('Project Information', { selector: '.ui-settings-card-header strong' })).toBeInTheDocument();
  expect(screen.getByText('Startup', { selector: '.ui-settings-card-header strong' })).toBeInTheDocument();
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

it('combines Android settings in one pane without planned or store navigation', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'Android' }));
  expect(screen.getByRole('textbox', { name: 'Android application ID' })).toBeInTheDocument();
  expect(screen.getByRole('spinbutton', { name: 'Android minimum SDK' })).toBeInTheDocument();
  expect(screen.queryByRole('treeitem', { name: /Planned|Google Play|Steam|Galaxy/ })).not.toBeInTheDocument();
  expect(screen.queryByRole('treeitem', { name: 'Application' })).not.toBeInTheDocument();
  expect(apply).not.toHaveBeenCalled();
});

it('displays existing project paths read-only without introducing relocation writes', async () => {
  snapshot.values['paths.content'] = 'GameContent';
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'Advanced' }));
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
  fireEvent.click(screen.getByRole('treeitem', { name: 'Cooking & Packaging' }));
  fireEvent.click(screen.getByRole('button', { name: 'Add profile' }));
  fireEvent.click(screen.getByText('profile', { selector: 'summary strong' }));
  expect(screen.getByRole('textbox', { name: 'Profile 1 ID' })).toHaveValue('profile');
  fireEvent.click(screen.getByRole('button', { name: 'Duplicate profile' }));
  fireEvent.click(screen.getByText('profile-copy', { selector: 'summary strong' }));
  expect(screen.getByRole('textbox', { name: 'Profile 2 ID' })).toHaveValue('profile-copy');
  fireEvent.click(screen.getByRole('treeitem', { name: 'Platforms' }));
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

it('limits navigation to two levels and groups multiple enabled targets by platform', () => {
  const categories = visibleProjectSettingsCategories(projectSettingsCatalog.categories, [
    { id: 'windows-x64-vulkan', enabled: true },
    { id: 'windows-x64-headless', enabled: true },
    { id: 'android-arm64-vulkan', enabled: false },
  ]);
  expect(categories.filter((entry) => !entry.parentId).length).toBeGreaterThan(5);
  for (const entry of categories) {
    if (entry.parentId) expect(categories.find((parent) => parent.id === entry.parentId)?.parentId).toBeUndefined();
  }
  expect(categories.filter((entry) => entry.platform === 'windows')).toHaveLength(1);
  expect(categories.some((entry) => entry.platform === 'android')).toBe(false);
  expect(categories.some((entry) => entry.id === 'platforms')).toBe(true);
});

it('uses the shared graphical preferences header and descriptive subtitles on every visible page', async () => {
  const { container } = render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  const pages = visibleProjectSettingsCategories(
    snapshot.catalog.categories,
    snapshot.values.targetPlatforms as { id: string; enabled: boolean }[],
  );
  for (const page of pages) {
    if (page.panelOnly) continue;
    fireEvent.click(screen.getByRole('treeitem', { name: page.label }));
    const header = container.querySelector('.ui-settings-header')!;
    expect(header.querySelector('h2')).toHaveTextContent(page.label);
    expect(header.querySelector('p')).toHaveTextContent(page.description!);
    expect(header.querySelector('img')).toHaveAttribute('src', expect.stringContaining('general-settings-header'));
  }
});

it('updates platform navigation from the draft without discarding disabled-platform settings', async () => {
  snapshot.values['platformSettings.android.applicationId'] = 'com.example.kept';
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'Platforms' }));
  fireEvent.click(screen.getByRole('switch', { name: 'android-arm64-vulkan' }));
  expect(screen.queryByRole('treeitem', { name: 'Android' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('switch', { name: 'android-arm64-vulkan' }));
  fireEvent.click(screen.getByRole('treeitem', { name: 'Android' }));
  expect(screen.getByRole('textbox', { name: 'Android application ID' })).toHaveValue('com.example.kept');
  expect(apply).not.toHaveBeenCalled();
});

it('falls back to the platform overview when the selected platform is disabled externally', async () => {
  render(<ProjectSettingsDialog onClose={vi.fn()} />);
  await screen.findByRole('textbox', { name: 'Project name' });
  fireEvent.click(screen.getByRole('treeitem', { name: 'Android' }));
  snapshot.revision = 'disabled';
  snapshot.values.targetPlatforms = [{ id: 'windows-x64-vulkan', enabled: true }];
  await act(async () => {
    notify();
  });
  expect(screen.queryByRole('treeitem', { name: 'Android' })).not.toBeInTheDocument();
  expect(screen.getByRole('switch', { name: 'windows-x64-vulkan' })).toBeInTheDocument();
  expect(screen.getByRole('treeitem', { name: 'Platforms' })).toHaveAttribute('aria-selected', 'true');
});
