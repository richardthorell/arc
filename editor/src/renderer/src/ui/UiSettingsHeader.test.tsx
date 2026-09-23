// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiSettingsHeader } from './UiSettingsHeader';

afterEach(cleanup);

describe('UiSettingsHeader', () => {
  it('renders a settings page title and optional subtitle with default artwork', () => {
    const { container } = render(<UiSettingsHeader subtitle="Startup and editor behavior." title="General" />);

    expect(screen.getByRole('heading', { name: 'General' })).toBeInTheDocument();
    expect(screen.getByText('Startup and editor behavior.')).toBeInTheDocument();
    expect(container.querySelector('.ui-settings-header-default-artwork')).toBeInTheDocument();
  });

  it('accepts custom graphical header artwork', () => {
    const { container } = render(
      <UiSettingsHeader background={<svg data-testid="custom-artwork" />} title="Viewport" />,
    );

    expect(screen.getByTestId('custom-artwork')).toBeInTheDocument();
    expect(container.querySelector('.ui-settings-header-default-artwork')).not.toBeInTheDocument();
  });
});
