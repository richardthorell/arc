// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiEditorToolbar, UiToolbarGroup, UiToolbarSeparator } from './UiEditorToolbar';

afterEach(cleanup);

describe('UiEditorToolbar', () => {
  it('provides stable left, center, and right toolbar regions', () => {
    render(
      <UiEditorToolbar
        aria-label="Test toolbar"
        className="domain-toolbar"
        left={<button>Save</button>}
        center={<UiToolbarGroup aria-label="Modes">Modes</UiToolbarGroup>}
        right={<UiToolbarSeparator />}
      />,
    );

    const toolbar = screen.getByRole('toolbar', { name: 'Test toolbar' });
    expect(toolbar).toHaveClass('main-toolbar', 'ui-editor-toolbar', 'domain-toolbar');
    expect(screen.getByText('Save').parentElement).toHaveAttribute('data-toolbar-region', 'left');
    expect(screen.getByLabelText('Modes')).toHaveClass('ui-toolbar-group', 'toolbar-group');
    expect(toolbar.querySelector('[data-toolbar-region="center"]')).toContainElement(screen.getByLabelText('Modes'));
    expect(toolbar.querySelector('[data-toolbar-region="right"] .toolbar-separator')).toBeInTheDocument();
  });

  it('allows embedded surfaces to override the default toolbar role', () => {
    render(<UiEditorToolbar aria-label="Embedded actions" role="presentation" left={<span>Save</span>} />);

    expect(screen.queryByRole('toolbar')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Embedded actions')).toHaveAttribute('role', 'presentation');
  });
});
