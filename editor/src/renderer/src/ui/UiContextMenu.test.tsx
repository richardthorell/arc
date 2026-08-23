// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiContextMenu, UiContextMenuItem } from './UiContextMenu';

afterEach(cleanup);

describe('UiContextMenu', () => {
  it('consumes wheel input without preventing native menu scrolling', () => {
    const parentWheel = vi.fn();
    render(
      <div onWheel={parentWheel}>
        <UiContextMenu aria-label="Node menu" x={20} y={30}>
          <div style={{ maxHeight: 40, overflowY: 'auto' }}>
            <UiContextMenuItem>Constant</UiContextMenuItem>
            <UiContextMenuItem>Vector 2</UiContextMenuItem>
          </div>
        </UiContextMenu>
      </div>,
    );

    const item = screen.getByRole('menuitem', { name: 'Constant' });
    const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY: 120 });
    item.dispatchEvent(event);

    expect(parentWheel).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
  });

  it('isolates pointer and context-menu input from its parent', () => {
    const parentPointer = vi.fn();
    const parentContext = vi.fn();
    render(
      <div onPointerDown={parentPointer} onContextMenu={parentContext}>
        <UiContextMenu aria-label="Node menu" x={20} y={30}>
          <UiContextMenuItem>Constant</UiContextMenuItem>
        </UiContextMenu>
      </div>,
    );

    const menu = screen.getByRole('menu', { name: 'Node menu' });
    fireEvent.pointerDown(menu);
    fireEvent.contextMenu(menu);

    expect(parentPointer).not.toHaveBeenCalled();
    expect(parentContext).not.toHaveBeenCalled();
  });
});
