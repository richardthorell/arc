// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { documentTypeIconCell, DocumentTypeIcon } from './DocumentTypeIcon';

describe('DocumentTypeIcon', () => {
  it('maps editor document and asset kinds to stable atlas cells', () => {
    expect(documentTypeIconCell('level')).toEqual([0, 0]);
    expect(documentTypeIconCell('scene')).toEqual([0, 0]);
    expect(documentTypeIconCell('material')).toEqual([1, 0]);
    expect(documentTypeIconCell('shader')).toEqual([2, 0]);
    expect(documentTypeIconCell('texture')).toEqual([3, 0]);
    expect(documentTypeIconCell('mesh')).toEqual([0, 1]);
    expect(documentTypeIconCell('prefab')).toEqual([1, 1]);
    expect(documentTypeIconCell('folder')).toEqual([2, 2]);
  });

  it('renders a clipped PNG atlas image at the requested cell', () => {
    const view = render(<DocumentTypeIcon kind="shader" size={20} title="Shader" />);
    const icon = view.getByRole('img', { name: 'Shader' });
    const atlas = icon.querySelector<HTMLImageElement>('[data-document-type-icon-image]');

    expect(icon).toHaveStyle({
      width: '20px',
      height: '20px',
      overflow: 'hidden',
      position: 'relative',
    });
    expect(icon).toHaveAttribute('data-document-type-icon', 'shader');
    expect(atlas).not.toBeNull();
    expect(atlas?.getAttribute('src')).toContain('document-type-icons-atlas.png');
    expect(atlas).toHaveStyle({
      width: '80px',
      height: '60px',
      left: '-40px',
      top: '0px',
    });
  });
});
