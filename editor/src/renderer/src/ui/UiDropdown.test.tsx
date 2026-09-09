// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { Monitor, Smartphone } from 'lucide-react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiDropdown } from './UiDropdown';

afterEach(cleanup);

describe('UiDropdown', () => {
  it('renders icon and text for the selected value and each option', () => {
    const onValueChange = vi.fn();
    render(
      <UiDropdown
        ariaLabel="Target platform"
        onValueChange={onValueChange}
        options={[
          { value: 'windows', label: 'Windows', icon: <Monitor data-testid="windows-icon" /> },
          { value: 'android', label: 'Android', icon: <Smartphone data-testid="android-icon" /> },
        ]}
        value="windows"
      />,
    );

    const trigger = screen.getByRole('button', { name: 'Target platform' });
    expect(trigger).toHaveClass('ui-dropdown-trigger');
    expect(trigger).toHaveTextContent('Windows');
    expect(screen.getByTestId('windows-icon')).toBeInTheDocument();

    fireEvent.click(trigger);
    expect(screen.getByRole('option', { name: /Android/ })).toBeInTheDocument();
    expect(screen.getByTestId('android-icon')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('option', { name: /Android/ }));
    expect(onValueChange).toHaveBeenCalledWith('android');
  });
});
