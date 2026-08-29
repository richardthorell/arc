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

  it('scales the full atlas and positions the requested cell by percentage', () => {
    const view = render(<DocumentTypeIcon kind="shader" size={20} title="Shader" />);
    const icon = view.getByRole('img', { name: 'Shader' });

    expect(icon).toHaveStyle({
      width: '20px',
      height: '20px',
      backgroundSize: '400% 300%',
      backgroundPosition: '66.66666666666666% 0%',
    });
    expect(icon.style.backgroundImage).toContain('document-type-icons-atlas.png');
  });
});
