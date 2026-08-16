// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { defaultArcThemeId } from '../themeRegistry';
import { UiLabThemePicker } from './UiLabThemePicker';

afterEach(() => {
  cleanup();
  delete document.documentElement.dataset.theme;
});

describe('UiLabThemePicker', () => {
  it('lists the registered editor themes and applies the selected theme to the document root', () => {
    render(<UiLabThemePicker />);

    const themePicker = screen.getByRole('combobox', { name: 'UI Lab theme' });
    expect(themePicker).toHaveValue(defaultArcThemeId);
    expect(screen.getAllByRole('option')).toHaveLength(1);
    expect(screen.getByRole('option', { name: 'Arc Dark' })).toBeInTheDocument();
    expect(document.documentElement.dataset.theme).toBe(defaultArcThemeId);
  });
});
