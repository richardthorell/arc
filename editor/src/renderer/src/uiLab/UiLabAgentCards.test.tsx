// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiLabAgentCards } from './UiLabAgentCards';

afterEach(cleanup);

describe('UiLabAgentCards', () => {
  it('represents every production AI response card type with deterministic fixtures', () => {
    const { container } = render(<UiLabAgentCards />);

    expect(screen.getByLabelText('AI response card gallery')).toBeVisible();
    for (const kind of ['task', 'tool', 'approval', 'diff', 'viewport', 'asset', 'error']) {
      expect(container.querySelector(`[data-activity-kind="${kind}"]`)).toBeInTheDocument();
    }
  });
});
