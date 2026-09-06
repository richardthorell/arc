// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SkeletonInspector } from './SkeletonInspector';

afterEach(cleanup);

describe('SkeletonInspector', () => {
  it('shows the hierarchy and selects a bone without creating an entity', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, error: '' });
    render(
      <SkeletonInspector
        command={command}
        skeleton={{
          name: 'Character',
          selectedJoint: 0,
          joints: [
            {
              index: 0,
              name: 'Hips',
              parent: -1,
              bindPosition: [0, 0, 0],
              bindRotation: [0, 0, 0, 1],
              bindScale: [1, 1, 1],
            },
            {
              index: 1,
              name: 'Spine',
              parent: 0,
              bindPosition: [0, 1, 0],
              bindRotation: [0, 0, 0, 1],
              bindScale: [1, 1, 1],
            },
          ],
        }}
      />,
    );

    expect(screen.getByRole('tree', { name: 'Skeleton hierarchy' })).toBeVisible();
    expect(screen.getAllByText('Hips')[0]).toBeVisible();
    fireEvent.click(screen.getByText('Spine'));
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('viewport.setSkeletonJoint', { viewportId: 'viewport-1', jointIndex: 1 }),
    );
    expect(screen.getByText('Selected Bone').nextSibling).toHaveTextContent('Spine');
    expect(screen.getByText('Parent').nextSibling).toHaveTextContent('Hips');
  });
});
