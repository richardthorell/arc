// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { MaterialNodeParameterControl } from './MaterialNodeParameterControl';

describe('MaterialNodeParameterControl', () => {
  it('uses the shared toggle and text input for parameter authoring', () => {
    const onEnabledChange = vi.fn();
    const onNameChange = vi.fn();

    render(
      <MaterialNodeParameterControl
        enabled={false}
        name="Roughness"
        onEnabledChange={onEnabledChange}
        onNameChange={onNameChange}
      />,
    );

    const toggle = screen.getByRole('switch', { name: 'Parameter' });
    const name = screen.getByRole('textbox', { name: 'Parameter name' });
    expect(toggle).toHaveAttribute('aria-checked', 'false');
    expect(name).toBeDisabled();

    fireEvent.click(toggle);
    expect(onEnabledChange).toHaveBeenCalledWith(true);
  });

  it('keeps the parameter name editable only while exposed', () => {
    const onNameChange = vi.fn();

    render(
      <MaterialNodeParameterControl enabled name="Base Color" onEnabledChange={vi.fn()} onNameChange={onNameChange} />,
    );

    const name = screen.getByRole('textbox', { name: 'Parameter name' });
    expect(name).toBeEnabled();
    fireEvent.change(name, { target: { value: 'Tint' } });
    expect(onNameChange).toHaveBeenCalledWith('Tint');
  });
});
