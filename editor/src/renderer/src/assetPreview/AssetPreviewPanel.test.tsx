// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { AssetPreviewPanel, AssetPreviewPlaceholder } from './AssetPreviewPanel';

describe('AssetPreviewPanel', () => {
  it('renders shared preview chrome, content, actions, and metadata', () => {
    render(
      <AssetPreviewPanel
        title="Material Preview"
        subtitle="Compiled asset thumbnail"
        actions={<button aria-label="Refresh preview">Refresh</button>}
        metadata={[
          { label: 'Mesh', value: 'Sphere' },
          { label: 'Environment', value: 'Studio' },
        ]}
      >
        <img alt="Material asset preview" src="data:image/png;base64,preview" />
      </AssetPreviewPanel>,
    );

    expect(screen.getByRole('region', { name: 'Material Preview' })).toBeInTheDocument();
    expect(screen.getByText('Compiled asset thumbnail')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh preview' })).toBeInTheDocument();
    expect(screen.getByAltText('Material asset preview')).toBeInTheDocument();
    expect(screen.getByText('Sphere')).toBeInTheDocument();
    expect(screen.getByText('Studio')).toBeInTheDocument();
  });

  it('renders the temporary preview widget without coupling to a renderer surface', () => {
    render(
      <AssetPreviewPanel title="Shader Preview">
        <AssetPreviewPlaceholder label="Shader preview" description="Native viewport integration pending." />
      </AssetPreviewPanel>,
    );

    expect(screen.getByText('Shader preview')).toBeInTheDocument();
    expect(screen.getByText('Native viewport integration pending.')).toBeInTheDocument();
  });
});
