// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiLabMaterialControls, UiLabMaterialPanels } from './UiLabMaterialGallery';

afterEach(cleanup);

describe('UI Lab material galleries', () => {
  it('includes material controls in the Controls gallery', () => {
    render(<UiLabMaterialControls />);

    expect(screen.getByRole('heading', { name: 'Material editor' })).toBeInTheDocument();
    expect(screen.getByText('Material parameters')).toBeInTheDocument();
    expect(screen.getByText('Material preview')).toBeInTheDocument();
    expect(screen.getByText('Material graph')).toBeInTheDocument();
    expect(screen.getByRole('application', { name: 'Material graph' })).toBeInTheDocument();
    expect(screen.getByLabelText('Material Preview')).toBeInTheDocument();
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
