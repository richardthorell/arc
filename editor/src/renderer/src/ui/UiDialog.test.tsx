// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiDialog } from './UiDialog';

afterEach(cleanup);

describe('UiDialog', () => {
  it('moves a modal dialog when its header is dragged', () => {
    render(
      <UiDialog title="Close project">
        <p>Unsaved changes will be lost.</p>
      </UiDialog>,
    );

    const dialog = screen.getByRole('dialog');
    const header = screen.getByText('Close project').closest('header')!;

    fireEvent.pointerDown(header, { button: 0, clientX: 100, clientY: 80 });
    fireEvent.pointerMove(window, { clientX: 135, clientY: 110 });

    expect(dialog).toHaveStyle('transform: translate3d(35px, 30px, 0)');
    expect(header).toHaveClass('is-draggable');

    fireEvent.pointerUp(window);
    expect(dialog).not.toHaveClass('is-dragging');
  });

  it('does not start a drag from the close button', () => {
    const onClose = vi.fn();
    render(<UiDialog title="Close project" onClose={onClose} />);

    const dialog = screen.getByRole('dialog');
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Close dialog' }), {
      button: 0,
      clientX: 100,
      clientY: 80,
    });
    fireEvent.pointerMove(window, { clientX: 135, clientY: 110 });

    expect(dialog).toHaveStyle('transform: translate3d(0px, 0px, 0)');
  });

  it('keeps preview dialogs fixed', () => {
    render(
      <UiDialog preview title="Preview dialog">
        Preview
      </UiDialog>,
    );

    const dialog = screen.getByRole('dialog');
    const header = screen.getByText('Preview dialog').closest('header')!;
    fireEvent.pointerDown(header, { button: 0, clientX: 100, clientY: 80 });
    fireEvent.pointerMove(window, { clientX: 135, clientY: 110 });

    expect(dialog).not.toHaveStyle('transform: translate3d(35px, 30px, 0)');
    expect(header).not.toHaveClass('is-draggable');
  });
});
