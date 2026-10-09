// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { MaterialGraphWorkspace } from './MaterialGraphWorkspace';

describe('MaterialGraphWorkspace', () => {
  it('provides the shared graph surface and an optional asset-specific sidebar', () => {
    const { container } = render(
      <MaterialGraphWorkspace
        graph={<div>Shared graph</div>}
        sidebar={<div>Asset-specific controls</div>}
        sidebarWidth={420}
      />,
    );

    expect(container.querySelector('.material-graph-workspace')).toBeInTheDocument();
    expect(screen.getByText('Shared graph')).toBeInTheDocument();
    expect(screen.getByText('Asset-specific controls')).toBeInTheDocument();
    expect(container.querySelector('.material-editor-sidebar')).toBeInTheDocument();
  });

  it('can host a graph without material-output chrome', () => {
    const { container } = render(<MaterialGraphWorkspace graph={<div>Function graph</div>} sidebar={null} />);

    expect(screen.getByText('Function graph')).toBeInTheDocument();
    expect(container.querySelector('.material-editor-sidebar')).not.toBeInTheDocument();
    expect(container.querySelector('.material-editor-divider')).not.toBeInTheDocument();
  });
});
