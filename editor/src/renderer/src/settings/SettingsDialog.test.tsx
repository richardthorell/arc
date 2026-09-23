// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EditorPreferencesDialog } from './EditorPreferencesDialog';
import { ProjectSettingsDialog } from './ProjectSettingsDialog';
import { SettingsDialog } from './SettingsDialog';
import { requestSettingsDialog, requestedSettingsDialogKind, resetSettingsDialogRequest } from './settingsDialogRoute';

beforeEach(() => {
  vi.stubGlobal('arc', {
    settings: {
      snapshot: vi.fn().mockResolvedValue({
        revision: 1,
        schema: [
          {
            key: 'renderer.defaultGrid',
            section: 'Renderer',
            label: 'Default Grid',
            description: 'Show the grid in new viewports.',
            type: 'boolean',
            defaultValue: true,
            scopes: ['user'],
          },
          {
            key: 'renderer.qualityTier',
            section: 'Renderer',
            label: 'Quality Tier',
            description: 'Renderer quality profile used by editor viewports.',
            type: 'enum',
            defaultValue: 'auto',
            options: ['auto', 'low', 'high'],
            scopes: ['user', 'project'],
          },
          {
            key: 'renderer.temporalHistoryWeight',
            section: 'Renderer',
            label: 'Temporal History Weight',
            description: 'Contribution retained from validated temporal history samples.',
            type: 'number',
            defaultValue: 0.9,
            minimum: 0,
            maximum: 1,
            step: 0.01,
            scopes: ['user', 'project'],
          },
          {
            key: 'renderer.projectOnly',
            section: 'Renderer',
            label: 'Project Only Setting',
            description: 'A setting owned exclusively by the project.',
            type: 'boolean',
            defaultValue: true,
            scopes: ['project'],
          },
        ],
        values: {
          'renderer.defaultGrid': true,
          'renderer.qualityTier': 'auto',
          'renderer.temporalHistoryWeight': 0.9,
          'renderer.projectOnly': true,
        },
        sources: {
          'renderer.defaultGrid': 'default',
          'renderer.qualityTier': 'default',
          'renderer.temporalHistoryWeight': 'default',
          'renderer.projectOnly': 'default',
        },
        restartRequired: ['renderer.qualityTier'],
      }),
      update: vi.fn(),
    },
    recovery: {
      snapshot: vi.fn().mockResolvedValue({ uncleanShutdown: false, generations: [] }),
      restore: vi.fn(),
      discard: vi.fn(),
    },
    extensions: {
      snapshot: vi.fn().mockResolvedValue({ extensions: [] }),
    },
  });
});

afterEach(() => {
  cleanup();
  resetSettingsDialogRequest();
  vi.unstubAllGlobals();
});

describe('EditorPreferencesDialog', () => {
  it('renders the searchable editor preferences shell', async () => {
    const onResetLayout = vi.fn();
    render(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={onResetLayout} />);

    expect(screen.getByRole('dialog', { name: 'Editor Preferences' })).toHaveAttribute('aria-modal', 'true');
    expect(screen.getByRole('tree', { name: 'Preference sections' })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Editing/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Viewport/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /AI/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Source Control/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Platforms & SDKs/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Windows/ })).toBeInTheDocument();
    expect(screen.queryByRole('combobox', { name: 'Settings scope' })).not.toBeInTheDocument();
    expect(screen.getByRole('searchbox', { name: 'Search preferences' })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Reset workbench layout' }));
    expect(onResetLayout).toHaveBeenCalledTimes(1);

    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));
  });

  it('shows only user-scoped settings and searches descriptor text', async () => {
    render(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByRole('searchbox', { name: 'Search preferences' }), {
      target: { value: 'default grid' },
    });
    expect(screen.getByRole('treeitem', { name: /Editing/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));

    expect(screen.getByRole('heading', { name: 'Viewport' })).toBeInTheDocument();
    expect(screen.getByText('Default Grid')).toBeInTheDocument();
    expect(screen.queryByText('Project Only Setting')).not.toBeInTheDocument();
    expect(screen.queryByRole('treeitem', { name: /Platforms & SDKs/ })).not.toBeInTheDocument();
  });

  it('writes preference edits only to user settings without showing success status', async () => {
    window.arc.settings.update = vi.fn().mockResolvedValue(null);
    render(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));
    fireEvent.click(screen.getByRole('switch', { name: 'Default Grid' }));

    await waitFor(() =>
      expect(window.arc.settings.update).toHaveBeenCalledWith('user', { 'renderer.defaultGrid': false }, 1),
    );
    expect(screen.queryByText(/updated in user settings/)).not.toBeInTheDocument();
  });

  it('shows restart requirements as separate warning metadata', async () => {
    render(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));

    expect(screen.getByText('Restart required')).toBeInTheDocument();
    expect(screen.getByText('Renderer quality profile used by editor viewports.')).toBeInTheDocument();
  });

  it('uses shared controls without stealing focus when callback props change', async () => {
    const { rerender } = render(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));

    const qualityTier = screen.getByRole('combobox', { name: 'Quality Tier' });
    fireEvent.click(qualityTier);
    expect(screen.getByRole('listbox')).toBeInTheDocument();

    const historyWeight = screen.getByRole('spinbutton', { name: 'Temporal History Weight' });
    historyWeight.focus();
    expect(historyWeight).toHaveFocus();

    rerender(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    expect(historyWeight).toHaveFocus();
  });

  it('renders OpenAI provider configuration without exposing a configured API key', async () => {
    window.arc.settings.snapshot = vi.fn().mockResolvedValue({
      revision: 4,
      schema: [
        {
          key: 'ai.openai.apiKey',
          section: 'AI Providers',
          label: 'API Key',
          description: 'OpenAI project API key.',
          type: 'string',
          format: 'secret',
          defaultValue: '',
          scopes: ['user'],
        },
        {
          key: 'ai.openai.model',
          section: 'AI Providers',
          label: 'Model',
          description: 'Default OpenAI model.',
          type: 'enum',
          defaultValue: 'gpt-6-sol',
          options: ['gpt-6-astra', 'gpt-6-sol', 'gpt-6-luna'],
          optionLabels: {
            'gpt-6-astra': 'GPT-6 Astra',
            'gpt-6-sol': 'GPT-6 Sol',
            'gpt-6-luna': 'GPT-6 Luna',
          },
          scopes: ['user'],
        },
        {
          key: 'ai.openai.reasoningEffort',
          section: 'AI Providers',
          label: 'Reasoning Effort',
          description: 'Default reasoning effort.',
          type: 'enum',
          defaultValue: 'medium',
          options: ['low', 'medium', 'high', 'xhigh', 'max'],
          scopes: ['user'],
        },
        {
          key: 'ai.openai.organizationId',
          section: 'AI Providers',
          label: 'Organization ID',
          description: 'Optional organization override.',
          type: 'string',
          defaultValue: '',
          scopes: ['user'],
        },
        {
          key: 'ai.openai.projectId',
          section: 'AI Providers',
          label: 'Project ID',
          description: 'Optional project override.',
          type: 'string',
          defaultValue: '',
          scopes: ['user'],
        },
        {
          key: 'ai.openai.storeResponses',
          section: 'AI Providers',
          label: 'Store Responses',
          description: 'Allow response retention.',
          type: 'boolean',
          defaultValue: false,
          scopes: ['user'],
        },
      ],
      values: {
        'ai.openai.apiKey': 'configured',
        'ai.openai.model': 'gpt-6-sol',
        'ai.openai.reasoningEffort': 'medium',
        'ai.openai.organizationId': '',
        'ai.openai.projectId': '',
        'ai.openai.storeResponses': false,
      },
      sources: {
        'ai.openai.apiKey': 'user',
        'ai.openai.model': 'default',
        'ai.openai.reasoningEffort': 'default',
        'ai.openai.organizationId': 'default',
        'ai.openai.projectId': 'default',
        'ai.openai.storeResponses': 'default',
      },
      restartRequired: [],
    });

    render(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);
    await waitFor(() => expect(window.arc.settings.snapshot).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole('treeitem', { name: /Providers/ }));

    expect(screen.getByText('OpenAI')).toBeInTheDocument();
    const apiKey = screen.getByLabelText('API Key');
    expect(apiKey).toHaveAttribute('type', 'password');
    expect(apiKey).toHaveAttribute('placeholder', 'Configured — enter a new key to replace');
    expect(apiKey).toHaveValue('');
    expect(screen.getByRole('combobox', { name: 'Model' })).toHaveTextContent('GPT-6 Sol');
    expect(screen.getByRole('combobox', { name: 'Reasoning Effort' })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Organization ID' })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Project ID' })).toBeInTheDocument();
    expect(screen.getByRole('switch', { name: 'Store Responses' })).not.toBeChecked();
    expect(screen.getByRole('button', { name: 'Remove API key from ARC' })).toBeEnabled();
  });

  it('shows framework pages that do not have registered preferences yet', () => {
    render(<EditorPreferencesDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);

    fireEvent.click(screen.getByRole('treeitem', { name: /Scene/ }));
    expect(screen.getByRole('heading', { name: 'Scene' })).toBeInTheDocument();
    expect(screen.getByText('No settings available')).toBeInTheDocument();
  });

  it('closes from the close button, Escape, and backdrop', () => {
    const onClose = vi.fn();
    const { rerender } = render(<EditorPreferencesDialog onClose={onClose} onResetLayout={vi.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Close dialog' }));
    expect(onClose).toHaveBeenCalledTimes(1);

    rerender(<EditorPreferencesDialog onClose={onClose} onResetLayout={vi.fn()} />);
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onClose).toHaveBeenCalledTimes(2);

    const backdrop = screen.getByRole('dialog', { name: 'Editor Preferences' }).parentElement;
    expect(backdrop).not.toBeNull();
    fireEvent.pointerDown(backdrop!);
    expect(onClose).toHaveBeenCalledTimes(3);
  });
});

describe('ProjectSettingsDialog', () => {
  it('uses the shared settings shell and remains empty for now', () => {
    render(<ProjectSettingsDialog onClose={vi.fn()} />);

    expect(screen.getByRole('dialog', { name: 'Project Settings' })).toBeInTheDocument();
    expect(screen.queryByRole('tree')).not.toBeInTheDocument();
    expect(window.arc.settings.snapshot).not.toHaveBeenCalled();
  });
});

describe('SettingsDialog', () => {
  it('routes the workbench settings host to project settings when requested', () => {
    requestSettingsDialog('projectSettings');
    const { unmount } = render(<SettingsDialog onClose={vi.fn()} onResetLayout={vi.fn()} />);

    expect(screen.getByRole('dialog', { name: 'Project Settings' })).toBeInTheDocument();
    unmount();
    expect(requestedSettingsDialogKind()).toBe('editorPreferences');
  });
});
