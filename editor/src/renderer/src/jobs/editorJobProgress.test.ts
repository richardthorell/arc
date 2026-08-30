import { describe, expect, it } from 'vitest';

import { beginEditorJob, getEditorJobProgress } from './editorJobProgress';

describe('editorJobProgress', () => {
  it('tracks a batch until the final job completes', () => {
    expect(getEditorJobProgress()).toBeNull();

    const first = beginEditorJob();
    const second = beginEditorJob();
    expect(getEditorJobProgress()).toEqual({ completed: 0, total: 2 });

    first.finish();
    expect(getEditorJobProgress()).toEqual({ completed: 1, total: 2 });

    second.finish();
    expect(getEditorJobProgress()).toBeNull();
  });
});
