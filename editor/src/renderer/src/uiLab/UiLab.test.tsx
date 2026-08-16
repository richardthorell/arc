// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiLab } from './UiLab';

afterEach(cleanup);

describe('UiLab', () => {
  it('renders the expanded editor control inventory', () => {
    render(<UiLab />);

    expect(screen.getByText('ARC UI Lab')).toBeInTheDocument();
    expect(screen.getByText('Buttons')).toBeInTheDocument();
    expect(screen.getByText('Text and form inputs')).toBeInTheDocument();
    expect(screen.getByText('Selection controls')).toBeInTheDocument();
    expect(screen.getByText('Inspector controls')).toBeInTheDocument();
    expect(screen.getByText('Asset references')).toBeInTheDocument();
    expect(screen.getByText('Navigation and containers')).toBeInTheDocument();
    expect(screen.getByText('Menus and popovers')).toBeInTheDocument();
    expect(screen.getByText('Feedback and states')).toBeInTheDocument();
    expect(screen.getByText('Window chrome')).toBeInTheDocument();
    expect(screen.getAllByText('ExampleComponent').length).toBeGreaterThan(0);

    expect(screen.getByRole('radio', { name: 'Static' })).toBeInTheDocument();
    expect(screen.getByRole('checkbox', { name: 'Realtime updates' })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Entity notes' })).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: 'Preview quality' })).toBeInTheDocument();
  });

  it('keeps gallery controls interactive', () => {
    render(<UiLab />);

    const entityName = screen.getByLabelText('Entity name');
    fireEvent.change(entityName, { target: { value: 'Bridge_02' } });
    expect(entityName).toHaveValue('Bridge_02');

    const frameLimit = screen.getByLabelText('Frame limit');
    fireEvent.change(frameLimit, { target: { value: '120' } });
    expect(frameLimit).toHaveValue(120);

    const movable = screen.getByRole('radio', { name: 'Movable' });
    fireEvent.click(movable);
    expect(movable).toBeChecked();

    const realtime = screen.getByRole('checkbox', { name: 'Realtime updates' });
    fireEvent.click(realtime);
    expect(realtime).not.toBeChecked();

    const previewQuality = screen.getByRole('slider', { name: 'Preview quality' });
    fireEvent.change(previewQuality, { target: { value: '88' } });
    expect(previewQuality).toHaveValue('88');

    fireEvent.click(screen.getByRole('button', { name: 'Global' }));
    expect(screen.getByRole('button', { name: 'Global' })).toHaveClass('is-active');
  });

  it('exposes the production titlebar menu in the window chrome preview', () => {
    render(<UiLab />);

    fireEvent.click(screen.getByRole('button', { name: 'File' }));
    expect(screen.getByRole('menuitem', { name: /^New Scene/ })).toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: /^Save Scene/ })).toBeInTheDocument();
  });
});
