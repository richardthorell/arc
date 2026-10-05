import { describe, expect, it } from 'vitest';

import { findNativeControlViolations } from './sharedControlMigration';

describe('shared control migration assertion', () => {
  it('reports native interactive controls', () => {
    expect(
      findNativeControlViolations(`
        <button>Save</button>
        <select><option>One</option></select>
        <input type="number" />
        <textarea />
      `).map(({ control }) => control),
    ).toEqual(['button', 'select', 'input', 'textarea']);
  });

  it('allows native file and hidden inputs at platform boundaries', () => {
    expect(
      findNativeControlViolations(`
        <input type="file" />
        <input type='hidden' />
      `),
    ).toEqual([]);
  });

  it('still rejects checkbox and radio inputs', () => {
    expect(
      findNativeControlViolations(`
        <input type="checkbox" />
        <input type="radio" />
      `).map(({ control }) => control),
    ).toEqual(['input']);
  });
});
