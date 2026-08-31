import { describe, expect, it } from 'vitest';

import { beginEditorJob, getEditorJobProgress } from './editorJobProgress';

describe('editorJobProgress', () => {
  it('tracks a named batch until the final job completes', () => {
    expect(getEditorJobProgress()).toBeNull();

    const first = beginEditorJob('Loading assets');
    const second = beginEditorJob('Loading workspace');
    expect(getEditorJobProgress()).toEqual({
      label: 'Loading assets',
      completed: 0,
      total: 2,
      indeterminate: false,
    });

    first.finish();
    expect(getEditorJobProgress()).toEqual({
      label: 'Loading workspace',
      completed: 1,
      total: 2,
      indeterminate: false,
    });

    second.finish();
    expect(getEditorJobProgress()).toBeNull();
  });

  it('lets foreground activities override background work and report their own progress', () => {
    const thumbnail = beginEditorJob('Generating thumbnails', { priority: 'background' });
    const startup = beginEditorJob('Opening scene', {
      priority: 'foreground',
      completed: 1,
      total: 4,
    });

    expect(getEditorJobProgress()).toEqual({
      label: 'Opening scene',
      completed: 1,
      total: 4,
      indeterminate: false,
    });

    startup.update({ label: 'Loading assets', completed: 2 });
    expect(getEditorJobProgress()).toEqual({
      label: 'Loading assets',
      completed: 2,
      total: 4,
      indeterminate: false,
    });

    startup.finish();
    expect(getEditorJobProgress()).toEqual({
      label: 'Generating thumbnails',
      completed: 0,
      total: 1,
      indeterminate: false,
    });

    thumbnail.finish();
    expect(getEditorJobProgress()).toBeNull();
  });

  it('supports indeterminate high-level work', () => {
    const importing = beginEditorJob('Importing assets', {
      priority: 'normal',
      indeterminate: true,
    });

    expect(getEditorJobProgress()).toEqual({
      label: 'Importing assets',
      completed: 0,
      total: 0,
      indeterminate: true,
    });

    importing.finish();
    expect(getEditorJobProgress()).toBeNull();
  });
});
