// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { CookWorkflowOptions } from '../../../common/projectWorkflowTypes';
import { BuildCreationDialog } from './BuildCreationDialog';
import type { BuildDialogAction } from './BuildToolbar';

const options: CookWorkflowOptions = {
  authority: 'project-sdk-version-1',
  profiles: [
    {
      id: 'android-debug',
      platform: 'android',
      targetId: 'android-arm64-vulkan',
      label: 'Android Debug · ASTC',
      editorBuildAvailable: false,
      editorBuildReason: 'Android editor packaged builds are not available yet',
    },
    {
      id: 'android-shipping',
      platform: 'android',
      targetId: 'android-arm64-vulkan',
      label: 'Android Shipping',
      editorBuildAvailable: false,
      editorBuildReason: 'Android editor packaged builds are not available yet',
    },
    {
      id: 'windows-shipping',
      platform: 'windows',
      targetId: 'windows-x64-vulkan',
      label: 'Windows Shipping',
      editorBuildAvailable: true,
    },
  ],
};

const cookOptions = vi.fn();
const executeWorkflow = vi.fn();
const execute = vi.fn();

beforeEach(() => {
  localStorage.clear();
  cookOptions.mockResolvedValue(options);
  executeWorkflow.mockResolvedValue({});
  execute.mockResolvedValue({});
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: { build: { cookOptions, executeWorkflow, execute } },
  });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

function open(action: BuildDialogAction = 'build', platform = 'android', projectKey = 'Game A') {
  const onClose = vi.fn();
  const onError = vi.fn();
  const view = render(
    <BuildCreationDialog
      action={action}
      projectKey={projectKey}
      hostPlatform="windows"
      preferredPlatform={platform}
      currentConfiguration="RelWithDebInfo"
      onClose={onClose}
      onError={onError}
    />,
  );
  return { onClose, onError, ...view };
}

describe('Build project configuration', () => {
  it('builds an available Windows package with authored profile and no device', async () => {
    localStorage.setItem(
      'arc.build.dialog.v1:Game A',
      JSON.stringify({ platform: 'windows', target: 'windows-x64-vulkan', profile: 'windows-shipping' }),
    );
    const { onClose } = open('build', 'windows');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Build' })).toBeEnabled());
    expect(screen.getByRole('button', { name: 'Build platform' })).toHaveTextContent('Windows');
    expect(
      screen.getByRole('button', { name: 'Build platform' }).querySelector('[data-platform-icon="windows"]'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Build target' })).toHaveTextContent('windows-x64-vulkan');
    expect(screen.getByRole('button', { name: 'Build profile' })).toHaveTextContent('Windows Shipping');
    expect(screen.queryByRole('button', { name: /device/i })).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Build' }));
    await waitFor(() =>
      expect(executeWorkflow).toHaveBeenCalledWith(
        {
          action: 'build',
          profile: 'windows-shipping',
          targetId: 'windows-x64-vulkan',
          destination: 'none',
          sessionId: '',
        },
        options.authority,
      ),
    );
    expect(onClose).toHaveBeenCalledOnce();
    expect(execute).not.toHaveBeenCalled();
  });

  it('uses friendly platform names and the main toolbar brand icons', async () => {
    cookOptions.mockResolvedValueOnce({
      ...options,
      profiles: [
        ...options.profiles,
        { id: 'ios-debug', platform: 'ios', targetId: 'ios-arm64-metal', label: 'iOS Debug' },
      ],
    });
    open('cook');
    const platform = screen.getByRole('button', { name: 'Build platform' });
    await waitFor(() => expect(platform).toHaveTextContent('Android'));
    expect(platform.querySelector('[data-platform-icon="android"]')).toBeInTheDocument();

    fireEvent.click(platform);
    const android = screen.getByRole('option', { name: 'Android' });
    const windows = screen.getByRole('option', { name: 'Windows' });
    const ios = screen.getByRole('option', { name: 'iOS' });
    expect(android.querySelector('[data-platform-icon="android"]')).toBeInTheDocument();
    expect(windows.querySelector('[data-platform-icon="windows"]')).toBeInTheDocument();
    expect(ios.querySelector('[data-platform-icon="ios"]')).toBeInTheDocument();

    fireEvent.click(ios);
    expect(platform).toHaveTextContent('iOS');
    expect(platform.querySelector('[data-platform-icon="ios"]')).toBeInTheDocument();
  });

  it('does not show unnecessary intro or loading copy while profiles resolve', async () => {
    let resolveProfiles!: (result: CookWorkflowOptions) => void;
    cookOptions.mockImplementationOnce(
      () =>
        new Promise<CookWorkflowOptions>((resolve) => {
          resolveProfiles = resolve;
        }),
    );
    open('cook');
    expect(screen.queryByText(/Choose what to build/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Loading available project profiles/i)).not.toBeInTheDocument();

    await act(async () => resolveProfiles(options));
    expect(screen.getByRole('button', { name: 'Build platform' })).toHaveTextContent('Android');
    expect(screen.queryByText(/Loading available project profiles/i)).not.toBeInTheDocument();
  });

  it('shows why an Android build is unavailable without blocking Android cooking', async () => {
    open();
    expect(await screen.findByRole('alert')).toHaveTextContent('Android editor packaged builds are not available yet');
    expect(screen.getByRole('button', { name: 'Build' })).toBeDisabled();
    expect(executeWorkflow).not.toHaveBeenCalled();
  });

  it('preserves editor code actions as host-only and respects chosen configuration', async () => {
    open('rebuild', 'android');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Rebuild' })).toBeEnabled());
    expect(screen.getByRole('button', { name: 'Build platform' })).toHaveTextContent('Windows');
    expect(screen.getByRole('button', { name: 'Build target' })).toHaveTextContent('Editor module');
    expect(screen.getByRole('button', { name: 'Build profile' })).toHaveTextContent('RelWithDebInfo');
    fireEvent.click(screen.getByRole('button', { name: 'Rebuild' }));
    await waitFor(() => expect(execute).toHaveBeenCalledWith({ action: 'rebuild', configuration: 'RelWithDebInfo' }));
    expect(executeWorkflow).not.toHaveBeenCalled();
  });

  it('cooks authored content without compiling, packaging or selecting a device', async () => {
    open('cook');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cook Content' })).toBeEnabled());
    fireEvent.click(screen.getByRole('button', { name: 'Cook Content' }));
    await waitFor(() =>
      expect(executeWorkflow).toHaveBeenCalledWith(
        {
          action: 'cook',
          profile: 'android-debug',
          targetId: 'android-arm64-vulkan',
          destination: 'none',
          sessionId: '',
        },
        options.authority,
      ),
    );
  });

  it('starts cooking without any save control or document-save step', async () => {
    open('cook');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cook Content' })).toBeEnabled());
    expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
    expect(screen.queryByText(/save.*document/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Cook Content' }));
    await waitFor(() => expect(executeWorkflow).toHaveBeenCalledOnce());
  });

  it('does not dispatch if the dialog closes during asynchronous profile revalidation', async () => {
    let resolveProfiles!: (result: CookWorkflowOptions) => void;
    const { onClose, unmount } = open('cook');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cook Content' })).toBeEnabled());
    cookOptions.mockImplementationOnce(
      () =>
        new Promise<CookWorkflowOptions>((resolve) => {
          resolveProfiles = resolve;
        }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Cook Content' }));
    await waitFor(() => expect(cookOptions).toHaveBeenCalledTimes(2));
    unmount();
    await act(async () => resolveProfiles(options));
    expect(executeWorkflow).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('does not start a workflow if the project options disappear before dispatch', async () => {
    const { onClose } = open('cook');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cook Content' })).toBeEnabled());
    cookOptions.mockResolvedValueOnce(null);
    fireEvent.click(screen.getByRole('button', { name: 'Cook Content' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Project or SDK changed');
    expect(executeWorkflow).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('refuses stale SDK/project profiles and does not dispatch a stale plan', async () => {
    const { onClose } = open('cook');
    await waitFor(() => expect(screen.getByRole('button', { name: 'Cook Content' })).toBeEnabled());
    cookOptions.mockResolvedValue({ ...options, authority: 'changed-sdk' });
    fireEvent.click(screen.getByRole('button', { name: 'Cook Content' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Project or SDK changed');
    expect(executeWorkflow).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it('does not advertise missing cooker support as a working packaged build', async () => {
    cookOptions.mockResolvedValue({ ...options, unavailableReason: 'ARC cooker is unavailable in this SDK' });
    open();
    expect(await screen.findByRole('alert')).toHaveTextContent('ARC cooker is unavailable');
    expect(screen.getByRole('button', { name: 'Build' })).toBeDisabled();
    expect(executeWorkflow).not.toHaveBeenCalled();
  });
});
