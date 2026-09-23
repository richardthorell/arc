// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiSettingsHeader } from './UiSettingsHeader';

afterEach(cleanup);

describe('UiSettingsHeader', () => {
  it('renders a settings page title and optional subtitle', () => {
    render(<UiSettingsHeader subtitle="Startup and editor behavior." title="General" />);

    expect(screen.getByRole('heading', { name: 'General' })).toBeInTheDocument();
    expect(screen.getByText('Startup and editor behavior.')).toBeInTheDocument();
  });
});
