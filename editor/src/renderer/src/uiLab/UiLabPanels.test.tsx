// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { panelRegistry } from '../app/panelRegistry';
import { panelSettingsFixture, panelSourceControlFixture } from './UiLabPanelFixtures';
import { UiLabPanels } from './UiLabPanels';
import { UiLabWindow } from './UiLabWindow';

const originalArc = window.arc;

beforeEach(() => {
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      getStartupState: vi.fn().mockResolvedValue({ engineHostConnected: false, viewportMode: 'streamed' }),
      assetSources: { list: vi.fn().mockResolvedValue([]) },
      projects: {
        readText: vi.fn().mockResolvedValue({
          path: 'Assets/Shaders/pbr_lit.hlsl',
          text: 'float4 PSMain() : SV_Target { return float4(1, 1, 1, 1); }',
          modifiedAt: '2026-08-15T23:17:00-07:00',
        }),
        writeText: vi.fn().mockResolvedValue({ succeeded: true }),
      },
      sourceControl: {
        snapshot: vi.fn().mockResolvedValue(panelSourceControlFixture),
        diff: vi.fn().mockResolvedValue({ succeeded: true, output: '', error: '' }),
        pull: vi.fn().mockResolvedValue({ succeeded: true, output: '', error: '' }),
        push: vi.fn().mockResolvedValue({ succeeded: true, output: '', error: '' }),
        stage: vi.fn().mockResolvedValue({ succeeded: true, output: '', error: '' }),
        unstage: vi.fn().mockResolvedValue({ succeeded: true, output: '', error: '' }),
        discard: vi.fn().mockResolvedValue({ succeeded: true, output: '', error: '' }),
        commit: vi.fn().mockResolvedValue({ succeeded: true, output: '', error: '' }),
      },
      settings: {
        snapshot: vi.fn().mockResolvedValue(panelSettingsFixture),
        update: vi.fn().mockResolvedValue(panelSettingsFixture),
      },
      recovery: { snapshot: vi.fn().mockResolvedValue(null) },
      extensions: { snapshot: vi.fn().mockResolvedValue(null) },
      viewport: undefined,
    },
  });
});

afterEach(() => {
  cleanup();
  Object.defineProperty(window, 'arc', { configurable: true, value: originalArc });
});

describe('UiLabPanels', () => {
  it('renders a slot for every registered editor panel', () => {
    const { container } = render(<UiLabPanels />);
    const previews = Array.from(container.querySelectorAll<HTMLElement>('[data-panel-id]'));
    const ids = previews.map((preview) => preview.dataset.panelId);

    expect(previews).toHaveLength(Object.keys(panelRegistry).length);
    expect(ids).toEqual(expect.arrayContaining(Object.keys(panelRegistry)));
    expect(screen.getByText('16 registered panels')).toBeInTheDocument();
  });

  it('mounts the production viewport, hierarchy, inspector, and content browser', () => {
    const { container } = render(<UiLabPanels />);

    expect(container.querySelector('[data-panel-id="viewport"] .arc-viewport-shell')).toBeInTheDocument();
    expect(container.querySelector('[data-panel-id="hierarchy"] .explorer-view')).toBeInTheDocument();
    expect(container.querySelector('[data-panel-id="inspector"] .data-inspector')).toBeInTheDocument();
    expect(container.querySelector('[data-panel-id="contentBrowser"] .content-browser-v2')).toBeInTheDocument();
    expect(screen.getByLabelText('Search hierarchy')).toBeInTheDocument();
    expect(screen.getByLabelText('Search assets')).toBeInTheDocument();
  });

  it('does not invent an Asset Explorer clone while it is still private to Workbench', () => {
    render(<UiLabPanels />);

    expect(screen.getByText('Assets is currently private to Workbench.tsx')).toBeInTheDocument();
    expect(screen.getByText(/does not duplicate its markup/i)).toBeInTheDocument();
  });
});

describe('UiLabWindow pages', () => {
  it('switches between control and panel pages from the top-level tabs', () => {
    render(<UiLabWindow />);

    expect(screen.getByText('ARC UI Lab')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Panels' }));
    expect(screen.getByText('Panel Lab')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Controls' }));
    expect(screen.getByText('ARC UI Lab')).toBeInTheDocument();
  });
});
