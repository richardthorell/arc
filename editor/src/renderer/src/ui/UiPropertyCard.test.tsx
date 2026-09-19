// @vitest-environment jsdom
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { UiPropertyCard } from './UiPropertyCard';

describe('UiPropertyCard', () => {
  it('renders data-driven fields through shared panel card rows', () => {
    const { container } = render(
      <UiPropertyCard
        expandable={false}
        fields={[
          { id: 'quality', label: 'Quality', description: 'Rendering quality', control: <button>High</button> },
          { id: 'section', label: 'Advanced', fullWidth: true },
        ]}
        title="Rendering"
      />,
    );

    expect(screen.getByText('Rendering')).toBeTruthy();
    expect(screen.getByText('Quality')).toBeTruthy();
    expect(screen.getByText('Rendering quality')).toBeTruthy();
    expect(screen.getByText('High')).toBeTruthy();
    expect(screen.getByText('Advanced')).toBeTruthy();
    expect(container.querySelectorAll('.ui-panel-card-row')).toHaveLength(2);
    expect(container.querySelector('.ui-panel-card-row.is-full-width')).not.toBeNull();
  });
});
