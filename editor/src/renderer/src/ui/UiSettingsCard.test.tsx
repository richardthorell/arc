// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiSettingsCard } from './UiSettingsCard';

afterEach(cleanup);

describe('UiSettingsCard', () => {
  it('renders optional header metadata and arbitrary settings content', () => {
    render(
      <UiSettingsCard icon={<span data-testid="icon" />} subtitle="Optional subtitle" title="Appearance">
        <button type="button">Theme</button>
      </UiSettingsCard>,
    );

    expect(screen.getByText('Appearance')).toBeInTheDocument();
    expect(screen.getByText('Optional subtitle')).toBeInTheDocument();
    expect(screen.getByTestId('icon')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Theme' })).toBeInTheDocument();
  });
});
