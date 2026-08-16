// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiLab } from './UiLab';

afterEach(cleanup);

describe('UiLab', () => {
  it('renders the production control families in isolation', () => {
    render(<UiLab />);

    expect(screen.getByText('ARC UI Lab')).toBeInTheDocument();
    expect(screen.getByText('Buttons')).toBeInTheDocument();
    expect(screen.getByText('Inspector controls')).toBeInTheDocument();
    expect(screen.getByText('Asset references')).toBeInTheDocument();
    expect(screen.getByText('Navigation and containers')).toBeInTheDocument();
    expect(screen.getByText('ExampleComponent')).toBeInTheDocument();
  });

  it('keeps gallery controls interactive', () => {
    render(<UiLab />);

    const entityName = screen.getByLabelText('Entity name');
    fireEvent.change(entityName, { target: { value: 'Bridge_02' } });
    expect(entityName).toHaveValue('Bridge_02');

    const frameLimit = screen.getByLabelText('Frame limit');
    fireEvent.change(frameLimit, { target: { value: '120' } });
    expect(frameLimit).toHaveValue(120);

    fireEvent.click(screen.getByRole('button', { name: 'Global' }));
    expect(screen.getByRole('button', { name: 'Global' })).toHaveClass('is-active');
  });
});
