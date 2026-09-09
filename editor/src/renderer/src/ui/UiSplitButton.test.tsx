// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { Hammer, RefreshCw } from 'lucide-react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiSplitButton } from './UiSplitButton';

afterEach(cleanup);

describe('UiSplitButton', () => {
  it('keeps the primary action separate from the dropdown actions', () => {
    const onClick = vi.fn();
    const onOptionSelect = vi.fn();
    render(
      <UiSplitButton
        icon={<Hammer />}
        label="Build"
        menuAriaLabel="Build actions"
        onClick={onClick}
        onOptionSelect={onOptionSelect}
        options={[
          { value: 'build', label: 'Build', icon: <Hammer /> },
          { value: 'rebuild', label: 'Rebuild', icon: <RefreshCw /> },
        ]}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Build' }));
    expect(onClick).toHaveBeenCalledOnce();
    expect(onOptionSelect).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Build actions' }));
    expect(screen.getByRole('menu')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('menuitem', { name: /Rebuild/ }));
    expect(onOptionSelect).toHaveBeenCalledWith('rebuild');
  });
});
