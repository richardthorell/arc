// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiLabMaterialControls, UiLabMaterialPanels } from './UiLabMaterialGallery';

afterEach(cleanup);

describe('UI Lab material galleries', () => {
  it('shows a representative material node in the Controls gallery', () => {
    render(<UiLabMaterialControls />);

    expect(screen.getByRole('heading', { name: 'Material nodes' })).toBeInTheDocument();
    expect(screen.getByText('Color node')).toBeInTheDocument();
    expect(screen.getByText('Color (RGBA)')).toBeInTheDocument();
    expect(screen.getByLabelText('Material node color picker')).toBeInTheDocument();
    expect(screen.getByLabelText('Material node parameter name')).toHaveValue('Base Color');
    expect(screen.queryByRole('application', { name: 'Material graph' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Material Preview')).not.toBeInTheDocument();

    const red = screen.getByLabelText('Material node R');
    fireEvent.change(red, { target: { value: '0.75' } });
    expect(red).toHaveValue(0.75);

    const parameter = screen.getByRole('checkbox', { name: 'Parameter' });
    fireEvent.click(parameter);
    expect(screen.queryByLabelText('Material node parameter name')).not.toBeInTheDocument();
  });

  it('includes non-dockable material editor surfaces in the Panels gallery', () => {
    render(<UiLabMaterialPanels />);

    expect(screen.getByText('Editor Surfaces')).toBeInTheDocument();
    expect(screen.getByText('Material Graph')).toBeInTheDocument();
    expect(screen.getByText('Material Preview')).toBeInTheDocument();
    expect(screen.getByText('Material Parameters')).toBeInTheDocument();
    expect(screen.getByText('3 material surfaces')).toBeInTheDocument();
  });
});
