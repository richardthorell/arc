// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiLabStatusBar } from './UiLabStatusBar';

afterEach(cleanup);

describe('UiLabStatusBar', () => {
  it('shows the production job progress state at fifty percent', () => {
    render(<UiLabStatusBar />);

    expect(screen.getByText('Job (5 / 10)')).toBeInTheDocument();
    const progress = screen.getByRole('progressbar', { name: 'Editor jobs: 5 of 10 complete' });
    expect(progress).toHaveAttribute('aria-valuenow', '5');
    expect(progress.querySelector('.status-job-progress-fill')).toHaveStyle({ width: '50%' });
  });
});
