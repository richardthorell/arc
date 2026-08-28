// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiLabMaterialControls, UiLabMaterialPanels } from './UiLabMaterialGallery';

afterEach(cleanup);

describe('UI Lab material galleries', () => {
  it('shows representative material node cards in the Controls gallery', () => {
    const { container } = render(<UiLabMaterialControls />);

    expect(screen.getByRole('heading', { name: 'Material nodes' })).toBeInTheDocument();
    const materialNodes = container.querySelector('.ui-lab-material-node-row');
    expect(materialNodes).toBeInTheDocument();
    expect(within(materialNodes as HTMLElement).getByText('Color', { selector: '.ui-node-card-title' })).toBeInTheDocument();
    expect(screen.queryByText('Color (RGB)')).not.toBeInTheDocument();
    expect(screen.queryByText('Color (RGBA)')).not.toBeInTheDocument();
    expect(screen.getByText('Texture Sample')).toBeInTheDocument();
    expect(screen.getByText('Constant')).toBeInTheDocument();
    expect(screen.getByLabelText('Open Color color picker')).toBeInTheDocument();
    expect(screen.getByLabelText('Material node parameter name')).toHaveValue('Base Color');
    expect(screen.getByLabelText('Texture parameter name')).toHaveValue('Albedo Texture');
    expect(screen.getByLabelText('Constant parameter name')).toHaveValue('Roughness');
    expect(screen.getByLabelText('Constant value')).toHaveValue(0.45);
    expect(screen.queryByRole('application', { name: 'Material graph' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Material Preview')).not.toBeInTheDocument();

    const red = screen.getByLabelText('Color R');
    fireEvent.change(red, { target: { value: '0.75' } });
    fireEvent.blur(red);
    expect(red).toHaveValue('0.75');

    const constant = screen.getByLabelText('Constant value');
    fireEvent.change(constant, { target: { value: '0.7' } });
    expect(constant).toHaveValue(0.7);

    const texture = screen.getByLabelText('Texture sample asset');
    expect(texture).toHaveTextContent('T_Bark_Albedo');
    fireEvent.click(texture);
    expect(texture).toHaveTextContent('T_Moss_Albedo');

    const colorParameterName = screen.getByLabelText('Material node parameter name');
    const parameterToggles = screen.getAllByRole('checkbox');
    expect(colorParameterName).toBeEnabled();
    fireEvent.click(parameterToggles[0]);
    expect(colorParameterName).toBeDisabled();
    expect(colorParameterName).toHaveValue('Base Color');
    fireEvent.click(parameterToggles[0]);
    expect(colorParameterName).toBeEnabled();
    expect(colorParameterName).toHaveValue('Base Color');
  });

  it('includes non-dockable material editor surfaces in the Panels gallery', () => {
    render(<UiLabMaterialPanels />);

    expect(screen.getByText('Editor Surfaces')).toBeInTheDocument();
    expect(screen.getByText('Material Graph')).toBeInTheDocument();
    expect(screen.getAllByText('Material Preview')).toHaveLength(2);
    expect(screen.getByText('Material Parameters')).toBeInTheDocument();
    expect(screen.getByText('3 material surfaces')).toBeInTheDocument();
  });
});